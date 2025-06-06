import logging
import os
import shutil
from collections import OrderedDict

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
import numpy as np
#from Adaptive_mask.masking import MaskingHook
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from ipywidgets import interact, fixed
from sklearn.manifold import TSNE




logger = logging.getLogger(__name__)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt/n
    return rt


def create_loss_fn(args):
    if args.label_smoothing > 0:
        criterion = SmoothCrossEntropy(alpha=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion.to(args.device)


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)


def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.save_path, exist_ok=True)
    if finetune:
        name = f'{args.dataset}_finetune'
    else:
        name = args.dataset
    filename = f'{args.save_path}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best==1:
        shutil.copyfile(filename, f'{args.save_path}/{args.dataset}_best.pth.tar')
        print("Saved checkpoint--STEP: ",state['step'])


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum =self.sum+ val * n
        self.count =self.count+ n
        self.avg = self.sum / self.count


def compute_l2(proto, x):

        diff = proto.unsqueeze(0) - x.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        return dist

def compute_prototype(args,x,labels):

        '''prototype = []

        sorted_la, indices = torch.sort(labels)
        sorted_x_la = x[indices].float()

        unique_labels, counts = torch.unique(sorted_la, return_counts=True)
        start_idx=0

        for i in range(args.way):
            if
            end_idx = start_idx + counts[i-1]-1
            prototype.append(torch.mean(sorted_x_la[start_idx:end_idx], keepdim=True, dim=0))
            start_idx = end_idx+1
            #print("proto:",prototype)


        prototype = torch.cat(prototype, dim=0)
        prototype = prototype.to(args.device)
        #print("prototype: ", prototype)'''
        # 初始化原型数量列表，初始值为 0
        d = x.shape[1]  # 获取词向量的维度
        prototypes_tensor = torch.full((args.way, d), float('nan'))

        # 计算每个类别的原型
        for label in torch.unique(labels):
            idx = torch.where(labels == label)[0]
            if len(idx) > 0:
                prototype = torch.mean(x[idx], dim=0)
                prototypes_tensor[label.item()] = prototype

        return prototypes_tensor.to(args.device)

def delet_nan_proto(proto_la,proto_un):
    has_nan_la = torch.isnan(proto_la).any().item()
    has_nan_un = torch.isnan(proto_un).any().item()
    if has_nan_la:
       nan_indices = torch.isnan(proto_la).any(dim=1)
       valid_indices = ~nan_indices  # 非 nan 的行索引
       proto_la = proto_la[valid_indices]
       proto_un = proto_un[valid_indices]
    if has_nan_un:
        nan_indices = torch.isnan(proto_un).any(dim=1)
        valid_indices = ~nan_indices  # 非 nan 的行索引
        proto_la = proto_la[valid_indices]
        proto_un = proto_un[valid_indices]

    return proto_la,proto_un


def compute_prototype_dist(args,data_tensor_un,hard_pe_un,data_tensor_l,labels):
    proto_la=compute_prototype(args,data_tensor_l,labels)
    proto_un=compute_prototype(args,data_tensor_un,hard_pe_un)

    proto_la,proto_un=delet_nan_proto(proto_la,proto_un)
    if len(proto_la)==0 and len(proto_un)==0:
        return 0
    else:
        proto_l_normalized = F.normalize(proto_la, p=2, dim=1)
        proto_un_normalized = F.normalize(proto_un, p=2, dim=1)

        dot_product = torch.sum(proto_l_normalized * proto_un_normalized, dim=1)
        distances = 1 - dot_product
        dis=torch.sum(distances)/len(distances)

        return dis
def adaptive_thres(args,max_probs,hard_pseudo_label):


    hard_pseudo_label_cpu = hard_pseudo_label.cpu()
    max_probs_cpu = max_probs.cpu()

    # 统计每个标签值对应的个数
    label_count = torch.zeros(args.way)
    for label in hard_pseudo_label_cpu:
        label_count[label.item()] += 1

    # 初始化用于存储每类对应的max_probs[i] > args.thres_class[hard_pseudo_label[i]]的个数
    result_list = torch.zeros(args.way)
    sum = torch.zeros(args.way)
    decrease = torch.zeros(args.way)
    scaling_factors = torch.zeros(args.way)
    new_thres_class=torch.zeros(args.way)

    # 统计满足条件的个数
    for i, label in enumerate(hard_pseudo_label_cpu):
        if max_probs_cpu[i] > args.thres_class[label.item()]:
            result_list[label.item()] += 1
            sum[label.item()] += max_probs_cpu[i]
    for i in range(args.way):
        if result_list[i] < label_count[i] / 2:
            decrease[i] = result_list[i]

    # 计算每类的缩放系数b

    max_count = torch.max(decrease)
    for i in range(len(result_list)):
        if result_list[i] == 0:
            result_list[i] += 1
    # scaling_factors = result_list.float() / denominator.float()
    # scaling_factors = [i / (2 - i) for i in range(scaling_factors)]
    #print("result_list", result_list)
    #print("decrease:", decrease)
    for i in range(args.way):
        if decrease[i] == 0:
            if label_count[i] == 0:
                scaling_factors[i] = 1
            else:
                new_thres_class[i] = sum[i] / result_list[i]
                scaling_factors[i] = 2
        else:
            temp = result_list[i] / (max_count + 1)
            scaling_factors[i] = temp / (2 - temp)
            new_thres_class[i] = new_thres_class[i] * scaling_factors[i]

    return new_thres_class

def ada_dis_thres(args,dist_l,dist_un_sample):
    #un_dist = torch.zeros(args.way)


    # 转换为 numpy 数组
    distances_l = dist_l.cpu().detach().numpy()
    distances_un = dist_un_sample.cpu().detach().numpy()

    # 找到每个样本在A和B中到最近原型的距离以及分类结果
    min_distances_A = np.min(distances_un, axis=1)
    classifications_A = np.argmin(distances_un, axis=1)
    min_dist_A_tensor = torch.tensor(min_distances_A).to(args.device)

    min_distances_B = np.min(distances_l, axis=1)
    classifications_B = np.argmin(distances_l, axis=1)

    # 计算A和B中每个类别的平均距离
    #class_counts_A = np.bincount(classifications_A)
    #class_counts_B = np.bincount(classifications_B)
    class_counts_A = torch.zeros(args.way)
    class_counts_B = torch.zeros(args.way)
    for label in classifications_A:
        class_counts_A[label.item()] += 1
    for label in classifications_B:
        class_counts_B[label.item()] += 1

    #print("classifications_un",classifications_A)
    #print("classifications_l",classifications_B)

    #print("class_counts_un",class_counts_A)
    #print("class_counts_l",class_counts_B)

    avgA = np.zeros(args.way)  # 存储A中每个类别的平均距离
    avgB = np.zeros(args.way)  # 存储B中每个类别的平均距离

    for i in range(args.way):
        if class_counts_B[i] == 0 and class_counts_A[i] != 0:
            avgB[i] = 100.0
            avgA[i] = np.mean(min_distances_A[classifications_A == i])
        elif class_counts_A[i] == 0 and class_counts_B[i] != 0:
            avgA[i] = 100.0
            avgB[i] = np.mean(min_distances_B[classifications_B == i])

        elif class_counts_A[i] == 0 and class_counts_B[i] == 0:
            avgA[i] = 100.0
            avgB[i] = 100.0
        else:
            avgA[i] = np.mean(min_distances_A[classifications_A == i])
            avgB[i] = np.mean(min_distances_B[classifications_B == i])

    #print("A中每个类别的平均距离：", avgA)
    #print("B中每个类别的平均距离：", avgB)

    # 统计A中每个类别样本的距离
    numA_less_than_distB = {i: 0 for i in range(args.way)}

    # 计算numA_less_than_distB
    for i in range(len(min_distances_A)):
        if min_distances_A[i] < avgB[classifications_A[i]]:
            numA_less_than_distB[classifications_A[i]] += 1

    # 计算每个类别的阈值
    thresholds = np.zeros(args.way)
    for i in range(args.way):
        numA = numA_less_than_distB[i]
        total_A = class_counts_A[i]
        if total_A == 0:
            thresholds[i] = avgB[i]
        else:
            if numA >= total_A / 2:
                thresholds[i] = min(avgA[i], avgB[i])
            else:
                thresholds[i] = max(avgA[i], avgB[i])
    #thresholds = torch.tensor(list(thresholds.values()), device='cuda:0')
    thresholds = torch.tensor(thresholds)
    return thresholds, min_dist_A_tensor




def plot_embedding_3d(X, y, args, title):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    '''X['labels'] = y
    df1 = X[X['labels'] == 0]
    df11 = X[X['labels'] == 10]
    df12 = X[X['labels'] == 11]
    df13 = X[X['labels'] == 12]'''
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    #color = ['lightcoral', 'tan', 'greenyellow', 'burlywood', 'crimson', 'palegreen']
    color = ['crimson', 'midnightblue']
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=plt.cm.Set1(np.argmax(y[:], axis=1) / 10.0))
    ax = Axes3D(fig)
    for i in range(X.shape[0]):
        if y[i]==0:
            ax.text(X[i, 0], X[i, 1], X[i, 2],str(y[i]) ,c='cornflowerblue', fontdict={'weight': 'bold', 'size': 4})
        else:
          #print(y[i])
          ax.text(X[i, 0], X[i, 1], X[i, 2],str(y[i]) ,c=color[y[i]-7], fontdict={'weight': 'bold', 'size': 4})

    # plt.title(title)
    '''ax.scatter(df1[:, 0], df1[:, 1], df1[:, 2], c='cornflowerblue', label='cover')
    ax.scatter(df11[:, 0], df11[:, 1], df11[:, 2], c=color[3], label='Tina-coco-1bpw')
    ax.scatter(df12[:, 0], df12[:, 1], df12[:, 2], c=color[4], label='GANs-movie-2bpw')
    ax.scatter(df13[:, 0], df13[:, 1], df13[:, 2], c=color[5], label='LSTM-news-3bpw')'''
    #ax.legend(loc='best')
    elevs = [ 20, 30]
    azims = [0, 30, 45, 60,75]
    for i, theta1 in enumerate(elevs):
        for j, theta2 in enumerate(azims):
            ax.view_init(elev=theta1,  # 仰角
                         azim=theta2  # 方位角
                         )
            # ax.set_title( f' {args.embedding} Twitter 仰角：{theta1}  方位角：{theta2}')
            path = "./image/Movie/"+args.num_labeled + '_' + "M" + str(theta1) + str(theta2)
            plt.savefig(path)


def tsne(x, y, args):
    # fig=plt.figure(figsize=(5,5))
    tsne = TSNE(n_components=3, init='pca', random_state=0).fit(x)
    X_tsne_3d = tsne.fit_transform(x)
    ##y=float(y)
    #x_min, x_max = np.min(tsne, 0), np.max(tsne, 0)
    #embedded = tsne / (x_max - x_min)


    plot_embedding_3d(X_tsne_3d[:,0:3], y, args, "t-SNE 3D")

def draw2d(x,y, args):

    tsne = TSNE(n_components=2, init='pca', random_state=501)

    result = tsne.fit_transform(x)
    print('result.shape', result.shape)
    x_min, x_max = result.min(0), result.max(0)
    data = (result - x_min) / (x_max - x_min)
    #color = ['slateblue', 'darkgray', 'darkgray', 'darkgray', 'darkgray', 'darkgray', 'darkgray']
    #color = ['slateblue', 'mediumturquoise', 'darkgreen', 'darkgray', 'violet', 'darkgoldenrod', 'crimson']#15 16 17
    #color = ['slateblue', 'deeppink', 'mediumspringgreen', 'darkgray', 'violet', 'darkgoldenrod', 'crimson']#avg 15 16 17
    #color = ['mediumslateblue', 'gold', 'mediumspringgreen', 'darkgray', 'violet', 'darkgoldenrod', 'crimson']
    color = ['crimson', 'midnightblue']
    plt.xlim((-80, 80))
    plt.ylim((-80, 80))
    plt.figure(figsize=(5, 5))
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.scatter(data[i, 0], data[i, 1],
                        color=color[y[i]],
                        marker='^')
        else:
            plt.scatter(data[i, 0], data[i, 1],
                 color=color[y[i]],
                 marker='.')

    plt.tick_params(axis='both', which='major', labelsize=10)
    #plt.rc('legend', fontsize=5)  # 图例字体大小
    #plt.legend()

    # plt.title('image')
    path="./image/"+str(args.num_labeled) + '_' + args.dataset+'_'+'draw_best'
    plt.savefig(path)

