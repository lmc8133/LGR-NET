export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29505 --nproc_per_node 8 train.py --batch_size 32 --aug_crop --aug_scale --aug_translate --bert_enc_num 12  --vl_dec_layers 6 --cls_num 1 --swin_checkpoint checkpoints/gref_umd_latest.pth --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_umd