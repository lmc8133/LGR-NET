import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from utils.config import Config
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, evaluate

# torch.distributed.init_process_group(backend="nccl")

import torch.distributed as dist

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=0., type=float)
    parser.add_argument('--lr_visu_cnn', default=0., type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)
    
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")
    parser.add_argument('--use_knowledge', default = True, type = bool)
    parser.add_argument('--split_knowledge_query', default = False, type = bool)

    # Model parameters
    parser.add_argument('--model_name', type=str, default='TransVG',
                        help="Name of model to be exploited.")
    
    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--visu_enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--use_vl_type_embed', action='store_true',
                        help="If true, use vl_type embedding")
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')
    parser.add_argument('--return_intermediate_dec', default=True, type=bool,
                        help="whether or not use aux loss from all decoder layers")
    parser.add_argument('--cls_num', default=1, type=int, help="Number of regression tokens used for box prediction in decoder")                                        
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=1, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--other_loss_coefs', default={}, type=float)

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--max_knowledge_len', default=120, type=int,
                        help='maximum SK knowledge length per batch')
    
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # QRNet parameters
    parser.add_argument('--soft_fpn', default='NoFpnSoftDownSample',type=str)
    parser.add_argument('--disable_spatial', action='store_true',
                        help="If true, use amp training")
    parser.add_argument('--disable_channel', action='store_true',
                        help="If true, use amp training")
    parser.add_argument('--swin_checkpoint', default='../QRNet/checkpoints/mask_rcnn_swin_small_patch4_window7.pth', type=str, help='QRNet checkpoint')
    parser.add_argument('--lr_visu_swin', default=1e-5, type=float)
    parser.add_argument('--lr_visu_fpn', default=1e-5, type=float)

    # evalutaion options
    parser.add_argument('--eval_set', default='test', type=str)
    parser.add_argument('--eval_model', default='', type=str)

    parser.add_argument('--config', type=str, help='Path to the configure file.')
    parser.add_argument('--model_config')
    return parser


def main(args):
    # for QRNet
    args.use_channel=not args.disable_channel
    args.use_spatial=not args.disable_spatial
    
    utils.init_distributed_mode(args)
    print(args)
    # dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

    # args.distributed=False
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    # print('here')
    # print(dataset_test.images)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)
    
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    batch_sampler_test = torch.utils.data.BatchSampler(
        sampler_test, args.batch_size, drop_last=False)
    if args.dataset == 'SK' and args.use_knowledge and args.split_knowledge_query:
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False,
                                    collate_fn=utils.collate_fn_sk, num_workers=args.num_workers)
    else:
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    # output log
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_log.txt").open("a") as f:
            f.write(str(args) + "\n")
    
    start_time = time.time()
    
    # perform evaluation
    accuracy, predxyxy, gtxyxy, iou = evaluate(model, data_loader_test, device)
    
    print(accuracy)
    print(predxyxy.shape)
    print(gtxyxy.shape)
    print(iou.shape)
    # print(query.shape)
    if args.eval_set=='testB':
        torch.save(predxyxy, output_dir / 'pred_coordinate_testB.pth')
        torch.save(gtxyxy, output_dir / 'gt_coordinate_testB.pth')
        torch.save(iou, output_dir/ 'iouB.pth')
    else:
        torch.save(predxyxy, output_dir / 'pred_coordinate_test.pth')
        torch.save(gtxyxy, output_dir / 'gt_coordinate_test.pth')
        torch.save(iou, output_dir/ 'iou.pth')
    # raise
    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        log_stats = {'test_model:': args.eval_model,
                    '%s_set_accuracy'%args.eval_set: accuracy,
                    }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
                with (output_dir / "eval_log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.config:
        cfg=Config(args.config)
        cfg.merge_to_args(args)    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
