import argparse
import logging
import math
import os
import time
import numpy as np
import torch
import random
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from sklearn.metrics import f1_score
import datetime
from torch.optim import AdamW

from tqdm import tqdm
from model import bert

from data import DATASET_GETTERS

from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict, compute_l2,compute_prototype,
                   compute_prototype_dist, adaptive_thres,ada_dis_thres,tsne,draw2d)

logger = logging.getLogger(__name__)

import warnings


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='AGNews',type=str,help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--dataset', default='AGNews', type=str,help='dataset name')
parser.add_argument('--num_labeled', type=int, default=100, help='number of labeled data')
parser.add_argument('--num_unlabeled', type=int, default=20000, help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total_steps', default=30000, type=int, help='number of total steps to run')
parser.add_argument('--eval_step', default=100, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--resize', default=32, type=int, help='resize text')
parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.0001, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.0001, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', default=False, action='store_true', help='only evaluate model on validation set')
parser.add_argument('--seed', default=2333, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda_u', default= 1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda_steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default= -1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default= -1,
                    help="For distributed training: clocal_rank")
parser.add_argument('--max_len', default=64, type=int, help='text_len')
parser.add_argument('--model', default='bert',type=str,help='model name')
parser.add_argument('--mode', default='train',type=str,help='mode name')
parser.add_argument("--gpu", type=int, default= 0,
                    help="gpu")
#add_argument_parser

parser.add_argument('--data_dic', default='', type=str, help='path to checkpoint')
parser.add_argument('--way', default=10, type=int, help='number of classes')
parser.add_argument('--balance_rate', default=0.5, type=int, help='the worstest situation for balance')
parser.add_argument('--thres_class', default=[], type=float, help='pseudo label threshold')
parser.add_argument('--result_path', default='', type=str, help='save_result')


def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True



def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']






def train_loop(args, labeled_loader, unlabeled_loader,dev_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler,dev_dataset,test_dataset,labeled_dataset):
    torch.cuda.empty_cache()
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")

    #initial threshold for each class
    global_theres_initial=1/args.way
    args.thres_class = [global_theres_initial for i in range(args.way)]
    print("initial thres: ", args.thres_class)

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    print("labeled_iter",labeled_iter)

    moving_dot_product = torch.empty(1).to(args.device)
    limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_dps = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            texts_l, targets = labeled_iter.next()
        except:
            if args.world_size > 1:
                labeled_epoch = labeled_epoch+1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            texts_l, targets = labeled_iter.next()
        texts_l = texts_l.squeeze(1)
        try:
            texts_uw, texts_us, _ = unlabeled_iter.next()

        except:
            if args.world_size > 1:
                unlabeled_epoch = unlabeled_epoch+1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            texts_uw, texts_us, _ = unlabeled_iter.next()
        texts_uw = texts_uw.squeeze(1)
        texts_us = texts_us.squeeze(1)
        data_time.update(time.time() - end)

        texts_l = texts_l.to(args.device)
        texts_uw = texts_uw.to(args.device)
        texts_us = texts_us.to(args.device)
        targets = targets.to(args.device)
        with amp.autocast(enabled=args.amp):
            batch_size = texts_l.shape[0]

            t_texts = torch.cat((texts_l, texts_uw, texts_us))
            #print("t_texts",t_texts)
            t_logits = teacher_model(t_texts)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)


            t_loss_l = criterion(t_logits_l, targets)
            soft_pseudo_label = torch.softmax(t_logits_uw.detach()/args.temperature, dim=-1)
            #print("soft_pseudo_label", soft_pseudo_label)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            #print("hard_pseudo_label", hard_pseudo_label)
            #print("t_u_logit", t_logits_uw)

            args.thres_class=adaptive_thres(args, max_probs, hard_pseudo_label)
            thresholds = [args.thres_class[label] for label in list(hard_pseudo_label)]
            threshold = torch.tensor(thresholds).to(args.device)
            mask = max_probs.ge(threshold).float()
            #print("MASK:", mask)

            '''proto_l = compute_prototype(args, t_logits_l, targets)
            dist_l = compute_l2(proto_l, t_logits_l)
            dist_un_sample = compute_l2(proto_l, t_logits_uw)
            pred_tun = torch.argmax(-dist_un_sample, dim=-1)
            args.thres_class, un_dis_sample=ada_dis_thres(args,dist_l,dist_un_sample)
            #print("args.thres_class:",args.thres_class)
            thresholds = [args.thres_class[label] for label in list(pred_tun)]
            threshold = torch.tensor(thresholds).to(args.device)
            mask = un_dis_sample.le(threshold).float()#.mean()
            #print("mask:",mask)'''

            #mask=max_probs.ge(0.95).float()


            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )

            #t_loss_u = torch.mean(-(-dist_un_sample * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask)

            weight_u = args.lambda_u * min(1., (step+1) / args.uda_steps)
            #weight_u= sum(adap_thres)/len(adap_thres)

            ts_loss_uda = t_loss_l + weight_u * t_loss_u#t_loss_uda
            s_texts = torch.cat((texts_l, texts_us))
            s_logits = student_model(s_texts)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            #print("targets:",targets)
            #print("hard_pseudo:",hard_pseudo_label)
            #print("s_logits_us:",s_logits_us)
            #print("t_logits_uw:",t_logits_uw)
            proto_dist=compute_prototype_dist(args,s_logits_us,hard_pseudo_label,s_logits_l,targets)
            #print("proto_dist:",proto_dist)
            #del s_logits
            s_loss_l_old = F.cross_entropy(s_logits_l, targets)
            st_loss_l_old = s_loss_l_old.detach()

            '''prototype_s_l = compute_prototype(args, s_logits_l, targets)
            #pred_s_l = -compute_l2(prototype_s_l, s_logits_l)

            has_nan_t_la = torch.isnan(prototype_s_l).any().item()
            distance_proto_s_l = []
            if has_nan_t_la:
                nan_indices = torch.isnan(prototype_s_l).any(dim=1)
                valid_indices = ~nan_indices
                prototype_s_l = prototype_s_l[valid_indices]
            for i in range(len(prototype_s_l)):
                for j in range(len(prototype_s_l)):
                    if i != j:
                        # 使用PyTorch中的cosine_similarity函数计算cosine距离
                        cosine_dist = 1 - F.cosine_similarity(prototype_s_l[i].unsqueeze(0),
                                                              prototype_s_l[j].unsqueeze(0))
                        distance_proto_s_l.append(cosine_dist.item())
            # 计算所有距离的总和
            #proto_dist_sl = torch.tensor(-sum(distance_proto_s_l)/len(distance_proto_s_l)).to(args.device)'''


            s_loss = criterion(s_logits_us, hard_pseudo_label)+st_loss_l_old+0.8*proto_dist#+proto_dist_sl
            #print("s_loss:", s_loss)
            #print("proto_dist",proto_dist)

        s_scaler.scale(s_loss).backward(retain_graph=True)

        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()


        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l1 = student_model(texts_l)

                #s_logits_us1=student_model(texts_us)
                #proto_dist1 = compute_prototype_dist(args, s_logits_us1, hard_pseudo_label, s_logits_l1, targets)
                #dot_product = proto_dist - proto_dist1

                prototype_s_l1 = compute_prototype(args, s_logits_l1, targets)
                has_nan_t_la = torch.isnan(prototype_s_l1).any().item()
                distance_proto_s_l = []
                if has_nan_t_la:
                    nan_indices = torch.isnan(prototype_s_l1).any(dim=1)
                    valid_indices = ~nan_indices
                    prototype_s_l1 = prototype_s_l1[valid_indices]
                for i in range(len(prototype_s_l1)):
                    for j in range(len(prototype_s_l1)):
                        if i != j:
                            # 使用PyTorch中的cosine_similarity函数计算cosine距离
                            cosine_dist = 1 - F.cosine_similarity(prototype_s_l1[i].unsqueeze(0),
                                                                  prototype_s_l1[j].unsqueeze(0))
                            distance_proto_s_l.append(cosine_dist.item())
                # 计算所有距离的总和
                # proto_dist_sl = torch.tensor(-sum(distance_proto_s_l)/len(distance_proto_s_l)).to(args.device)
                proto_dist_sl = torch.tensor(sum(distance_proto_s_l)).to(args.device)

            #s_loss_l_new = F.cross_entropy(s_logits_l1.detach(), targets)
            #dot_product = st_loss_l_old - s_loss_l_new
            #print("dot_product:",dot_product)
            #print("st_loss_l_old:", st_loss_l_old)
            #print("s_loss_l_new:", s_loss_l_new)
            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            #print("proto_dist_sl:", proto_dist_sl)
            t_loss_dps =  F.cross_entropy(t_logits_us, hard_pseudo_label)/(proto_dist_sl+1)#dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            t_loss = t_loss_l + ts_loss_uda + t_loss_dps

        t_scaler.scale(t_loss).backward(retain_graph=True)
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()
        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_dps = reduce_tensor(t_loss_dps.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_dps.update(t_loss_dps.item())
        mean_mask.update(mask.mean().item())


        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f" S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)

        args.num_eval = step//args.eval_step
        if (step+1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_dps", t_losses_dps.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                test_model = avg_student_model if avg_student_model is not None else teacher_model
                test_loss, top1 = evaluate(args, dev_loader, s_logits_l1,targets,test_model, criterion)
                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)


                #is_best = top1 >= args.best_top1
                if top1>=args.best_top1:
                    args.best_top1 = top1
                    is_best=1
                else:
                    is_best = 0


                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if args.local_rank in [-1, 0]:
        args.writer.add_scalar("result/test_acc@1", args.best_top1)
    return



def evaluate(args, test_loader, labeled_data_tensor,labels,model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    #print("test_iter", test_iter)
    #print("test_loader", test_loader)
    with torch.no_grad():
        end = time.time()
        # lbs = []

        for step, (texts, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = texts.shape[0]
            texts = texts.to(args.device)
            targets = targets.to(args.device)
            target = targets.detach().cpu().numpy().tolist()
            texts = texts.squeeze(1)
            #print("texts",texts)
            #print("targets",targets)
            #print("step",step)

            with amp.autocast(enabled=args.amp):
                outputs = model(texts)

                #outputs = F.softmax(outputs, dim=1)
                op = outputs.detach().cpu().numpy().tolist()
                prototype = compute_prototype(args,labeled_data_tensor,labels)
                pred =-compute_l2(prototype, outputs)
                for output in op:

                    lb_value = max(output)

                #loss = criterion(outputs, targets)
                loss = criterion(pred, targets)

            #acc1 = accuracy(outputs, targets)
            acc1 = accuracy(pred, targets)

            losses.update(loss.item(), batch_size)



            top1.update(acc1[0].item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}")

            test_iter.close()
        return losses.avg, top1.avg


def inference_fn(args,model, dataloader,labeled_loader):
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    '''for batch in labeled_loader:
            batch_texts_l, batch_targets = batch[0].to(args.device), batch[1].to(args.device)
            if num==0:
                x=batch_texts_l
                y=batch_targets
            elif num < 20:
                x =torch.cat((x,batch_texts_l),0)'''


    texts_l_proto=[]
    labels_proto=[]
    for batch in labeled_loader:
        batch_texts_l, batch_targets = batch[0].to(args.device), batch[1].to(args.device)
        with torch.no_grad():
            texts_l = model(batch_texts_l)
            labels = batch_targets
        texts_l_proto.append(texts_l)
        labels_proto.append(labels)
    # 将列表转换为 torch.tensor 形式（如果需要的话）
    texts_l_proto = torch.cat(texts_l_proto, dim=0)
    labels_proto = torch.cat(labels_proto, dim=0)
    #print("texts_l_proto", texts_l_proto)
    #print("targets", labels_proto)
    prototype = compute_prototype(args, texts_l_proto, labels_proto)
    num=0

    for data in dataloader:
        inputs, targets = data[0].to(args.device), data[1].to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
            labels = targets.data.cpu().numpy()
            #predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            pred_logi = -compute_l2(prototype, outputs)
            predic=torch.argmax(pred_logi, dim=1).detach().cpu().numpy()
            #print("predic",predic)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            '''if args.evaluate!=0:
            if num==0:
                print("----START DRAWING----")
                x=outputs
                y=targets
            else:
                x=torch.cat((x,outputs),0)
                y = torch.cat((y, targets), 0)'''
            num=num+1
        # preds.append(outputs.sigmoid().detach().cpu().numpy())
    '''if args.evaluate != 0:
        draw2d(x.cpu().detach().numpy(), y.cpu().detach().numpy(), args)
        print("******plot success*****")'''
    mif1 = f1_score(labels_all, predict_all, average='micro')  # 调用并输出计算的值
    maf1 = f1_score(labels_all, predict_all, average='macro')
    report = metrics.classification_report(labels_all, predict_all, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return report, confusion, mif1,maf1

def create_model(args):
    model = bert.Model(args)
    return model
def main():
    args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top5 = 0.

    args.device = torch.device('cuda', args.gpu)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"result/{args.name}")

    if args.seed is not None:
        setup_seed(args.seed)


    labeled_dataset, unlabeled_dataset, dev_dataset, test_dataset = DATASET_GETTERS[args.dataset](args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True)
    try:
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler(unlabeled_dataset),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)
    except:
        pass
    dev_loader = DataLoader(dev_dataset,
                             sampler=SequentialSampler(dev_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)

    teacher_model =create_model(args)
    student_model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    teacher_model.to(args.device)
    student_model.to(args.device)
    avg_student_model = None


    criterion = create_loss_fn(args)

    t_optimizer = optim.SGD([{'params':teacher_model.encoder.parameters()},{'params':teacher_model.fc.parameters(),'lr':0.001}],
                            lr=args.teacher_lr,
                            momentum=args.momentum,
    #                        weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD([{'params':student_model.encoder.parameters()},{'params':student_model.fc.parameters(),'lr':0.001}],
                            lr=args.student_lr,
                            momentum=args.momentum,
     #                       weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    ''''t_optimizer = AdamW([{'params':teacher_model.encoder.parameters()},{'params':teacher_model.fc.parameters(),'lr':0.001}],
                        lr=args.teacher_lr,
                        betas=(0.9, 0.999),  # AdamW 默认的 betas 参数
                        weight_decay=args.weight_decay)
    s_optimizer = AdamW([{'params':student_model.encoder.parameters()},{'params':student_model.fc.parameters(),'lr':0.001}]
                        , lr=args.student_lr, betas=(0.9, 0.999),
                        weight_decay=args.weight_decay)'''


    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            # args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            # args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            if not (args.evaluate):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
            load_step=checkpoint['step']
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)


    if args.mode=='train' and args.evaluate==0:
        teacher_model.zero_grad()
        student_model.zero_grad()
        train_loop(args, labeled_loader, unlabeled_loader, dev_loader,
                teacher_model, student_model, avg_student_model, criterion,
                t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler,dev_dataset,test_dataset,labeled_dataset)
        report, confusion, mif1, maf1 = inference_fn(args, student_model, test_loader, labeled_loader)
        print(report, confusion, mif1, maf1)

    else:
        report, confusion, mif1, maf1 = inference_fn(args, student_model, test_loader, labeled_loader)
        print(report, confusion, mif1, maf1)

    total = (sum([param.nelement() for param in student_model.parameters()])+sum([param.nelement() for param in teacher_model.parameters()]))
    print("Number of parameter: %.2fM" % (total / 1e6))

    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_batch_result": report,
            "mif1": mif1,
            "maf1": maf1,
        }
        Task = args.dataset
        Labeled_samples=args.num_labeled
        unlabeled_samples=args.num_unlabeled
        total_steps=args.total_steps
        now_time = datetime.datetime.now()
        now_time =datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')

        with open(args.result_path, "a") as f:
            f.write("  domain: " + Task + "  Total steps: " + str(total_steps) + '\n')
            f.write("  "+now_time + "  Labeled_samples: " + str(Labeled_samples) + "  query: " + str(unlabeled_samples)+"  way: " + str(args.way)+ '\n')
            if args.resume:
                f.write("  CHECKPOINT: " + args.resume +  "  STEP:"+ str(load_step)+'\n')
            else:
                f.write("  The Lastest Model " + '\n')
            f.write("  test_batch_result: " + '\n')
            f.write(report + "   mif1: " + str(mif1) + "   maf1: " + str(maf1) + '\n')
            f.write('\r\n')
            f.write('\r\n')
            f.write('\r\n')
            f.write('\r\n')
        print("Test results are stored in the result.txt file!")

    return


if __name__ == '__main__':
    main()
