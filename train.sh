# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ReferItGame
# CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50_diou --resume ./outputs/referit_r50_diou/best_checkpoint.pth


# # RefCOCO
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco_r50 --resume ./outputs/refcoco_r50/checkpoint.pth


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco_plus_r50 --epochs 180 --lr_drop 120


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50


# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r50

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50 --epochs 90 --lr_drop 60 --resume ./outputs/referit_r50/checkpoint.pth
# torchrun --nproc_per_node 2 train.py --batch_size 32 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_umd_r50_diou
# CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50_diou --resume ./outputs/referit_r50_diou/best_checkpoint.pth
# torchrun train.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_umd_r50_diou --resume ./outputs/refcocog_umd_r50_diou/checkpoint.pth

# refcocog umd-split
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 train.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --visu_enc_layers 6 --vl_dec_layers 6  --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_umd_conditional_detr

# refcoco+
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --visu_enc_layers 6 --vl_dec_layers 6  --dataset unc+ --max_query_len 40 --output_dir outputs/refcoco+_conditional_detr_auxloss

# QRNet
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py --batch_size 32 --aug_crop --aug_scale --aug_translate --bert_enc_num 12  --vl_dec_layers 6  --config models/decoder_config/vltvg_decoder.py --dataset gref --max_query_len 40 --output_dir outputs/refcocog_google_vltvg_QRNet --swin_checkpoint ../QRNet/checkpoints/gref_latest.pth

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 train.py --batch_size 32 --aug_crop --aug_scale --aug_translate --bert_enc_num 12  --vl_dec_layers 6 --cls_num 1 --dataset gref_umd --max_query_len 40 --swin_checkpoint ../QRNet/checkpoints/gref_umd_latest.pth --output_dir outputs/refcocog_umd_papermethod_nocrossalign

# CUDA_VISIBLE_DEVICES=1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29505 --nproc_per_node 1 train.py --batch_size 32 --aug_crop --aug_scale --aug_translate --bert_enc_num 12  --vl_dec_layers 6 --cls_num 1 --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --dataset referit --max_query_len 40 --output_dir outputs/referit_papermethod_res101
CUDA_VISIBLE_DEVICES=1 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29505 --nproc_per_node 1 train.py --batch_size 32 --aug_crop --aug_scale --aug_translate --bert_enc_num 12  --vl_dec_layers 6 --cls_num 1 --swin_checkpoint ../QRNet/checkpoints/gref_umd_latest.pth --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_umd_papermethod_text_attn_image