# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
import pdb


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, max_norm: float = 0):
    use_amp=True    # for amp setting
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # pdb.set_trace()
        img_data, text_data, target = batch

        # copy to GPU
        img_data = img_data.to(device)
        if isinstance(text_data, tuple):
            text_data = (text_data[0].to(device), text_data[1].to(device))
        else:
            text_data = text_data.to(device)
        target = target.to(device)

        # model forward
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output, class_loss = model(img_data, text_data)
            # output = model(img_data, text_data)

            loss_dict = criterion(output, target)
            weight_dict=criterion.weight_dict
            loss_dict['loss_cls'] = class_loss
            # loss_dict = loss_utils.trans_vg_loss(output, target)  #origin loss
            # loss_dict = loss_utils.mytrans_vg_loss(output, target)  #diou loss
            # loss_dict = loss_utils.ciou_loss(output, target)

            losses = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v*weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)
        
        optimizer.zero_grad()
        
        # losses.backward()    # amp setting
        scaler.scale(losses).backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer.step()  # amp setting
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        if isinstance(text_data, tuple):
            text_data = (text_data[0].to(device), text_data[1].to(device))
        else:
            text_data = text_data.to(device)
        target = target.to(device)
        
        pred_boxes, class_loss = model(img_data, text_data)
        # pred_boxes = model(img_data, text_data)
        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes['pred_boxes'], target)
        
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    text=[]
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target = batch
        # batch_size = img_data.tensors.size(0)
        # copy to GPU
        text.append(text_data)
        img_data = img_data.to(device)
        if isinstance(text_data, tuple):
            text_data = (text_data[0].to(device), text_data[1].to(device))
        else:
            text_data = text_data.to(device)
        target = target.to(device)
        output, class_loss = model(img_data, text_data)
        # output = model(img_data, text_data)

        pred_box_list.append(output['pred_boxes'].cpu())          #xywh
        gt_box_list.append(target.cpu())            #xywh

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    # pdb.set_trace()
    accu_num, pred_boxes, gt_boxes, iou = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    
    return accuracy, pred_boxes, gt_boxes, iou
        