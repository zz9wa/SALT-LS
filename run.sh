data_dic=''
result_path=''


python main_temp.py \
--dataset 'ADG_n' \
--num_labeled 50 \
--num_unlabeled 10000 \
--batch-size 16 \
--max_len 64 \
--total_steps 30000 \
--data_dic $data_dic \
--way 2 \
--result_path $result_path \
--eval_step 100 \
--teacher_lr 0.0001
