# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=1


# ReferItGame
# python -m torch.distributed.launch --nproc_per_node=1 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./outputs/referit_r50/best_checkpoint.pth --output_dir ./outputs/referit_r50-test


# # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc --max_query_len 20 --eval_set testA --eval_model ../released_models/TransVG_unc.pth --output_dir ./outputs/refcoco_r50


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc+ --max_query_len 20 --eval_set testA --eval_model ../released_models/TransVG_unc+.pth --output_dir ./outputs/refcoco_plus_r50


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref --max_query_len 40 --eval_set val --eval_model ../released_models/TransVG_gref.pth --output_dir ./outputs/refcocog_gsplit_r50

# --rdzv_backend=c10d --rdzv_endpoint=localhost:29501
# # RefCOCOg u-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ../released_models/TransVG_gref_umd.pth --output_dir ./outputs/refcocog_usplit_r50
# CUDA_VISIBLE_DEVICES=0 torchrun myeval.py --batch_size 128 --bert_enc_num 12 --vl_dec_layers 6 --dataset gref_umd --max_query_len 40 --eval_set test --config models/decoder_config/vltvg_decoder.py --eval_model ./outputs/refcocog_umd_vltvg_QRNet/best_checkpoint.pth --output_dir ./outputs/refcocog_umd-test
CUDA_VISIBLE_DEVICES=1 torchrun myeval.py --batch_size 64 --bert_enc_num 12 --vl_dec_layers 6 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model outputs/refcocog_umd_papermethod_text_attn_image/best_checkpoint.pth --output_dir outputs/refcocog_umd_papermethod_text_attn_image
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --use_env myeval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./outputs/referit_r50_diou/best_checkpoint.pth --output_dir ./outputs/referit_r50-test
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --use_env myeval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./outputs/referit_r50_diou/best_checkpoint.pth --output_dir ./outputs/referit_r50-test