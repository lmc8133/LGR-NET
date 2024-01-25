from functools import partial

from einops.einops import rearrange,repeat
from torch import nn
import argparse
import os
import warnings

from mmcv import Config, DictAction
from mmcv.runner import  load_checkpoint
import sys

from mmdet.models import build_detector, build_backbone, build_neck
from icecream import ic
import torch
from math import sqrt

class QRNet(nn.Module):
    def __init__(self,args) -> None:
        super(QRNet,self).__init__()
        self.args=args
        self.flag=None
        config='models/swin_model/configs/my_config/simple_multimodal_fpn_config.py'
        self.flag='multi-modal'
        cfg_options=None

        checkpoint=args.swin_checkpoint
        cfg = Config.fromfile(config)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        cfg.model.neck.type=args.soft_fpn
        cfg.model.backbone.use_spatial=args.use_spatial
        cfg.model.backbone.use_channel=args.use_channel
        cfg.model.neck.use_spatial=args.use_spatial
        cfg.model.neck.use_channel=args.use_channel
    
        
        self.cfg=cfg
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        self.pretrained_parameter_name=checkpoint.keys()  
        
        self.num_channels=cfg.model['neck']['out_channels']
        self.backbone=model.backbone
        self.neck=model.neck
        self.rpn_head=model.rpn_head

    
    def forward(self,img,mask,text=None,extra:dict=None):
        import torch.nn.functional as F
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img,text) #img[bs, C, H, W](8,3,640,640) text[bs, D](8,768)

        x = self.neck(x,text)   #[[bs, C, H(160), W(160)], [bs, C, H/2, W/2], ..., [bs, C, H/16, W/16]]
        out_mask=[F.interpolate(mask[None].float(), size=_.shape[-2:]).to(torch.bool)[0] for _ in x]
    
        # flatten 
        shape=[_.shape[-2:] for _ in x] 
        x=[rearrange(_,'B C H W -> (H W) B C') for _ in x]
        out_mask=[rearrange(_,'B H W -> B (H W)') for _ in out_mask]
        return x,out_mask

# class QRResnet(nn.Module):
#     def __init__(self,args) -> None:
#         super(QRResnet,self).__init__()
#         self.args=args
#         self.flag=None
#         # config='models/swin_model/configs/my_config/simple_multimodal_fpn_config.py'
#         config='models/swin_model/configs/my_config/multi_modal_mask_rcnn_r50_fpn.py'
#         self.flag='multi-modal'
#         cfg_options=None

#         checkpoint=args.detr_model
#         cfg = Config.fromfile(config)
#         # if cfg_options is not None:
#         #     cfg.merge_from_dict(cfg_options)
#         # if cfg.get('custom_imports', None):
#         #     from mmcv.utils import import_modules_from_strings
#         #     import_modules_from_strings(**cfg['custom_imports'])
#         cfg.neck.type=args.soft_fpn
#         cfg.backbone.use_spatial=args.use_spatial
#         cfg.backbone.use_channel=args.use_channel
#         cfg.neck.use_spatial=args.use_spatial
#         cfg.neck.use_channel=args.use_channel
    
        
#         self.cfg=cfg
#         backbone = build_backbone(cfg.backbone)
#         neck = build_neck(cfg.neck)
#         checkpoint = load_checkpoint(backbone, checkpoint, map_location='cpu')
#         # self.pretrained_parameter_name=checkpoint.keys()  
        
#         self.num_channels=cfg['neck']['out_channels']
#         self.backbone=backbone
#         self.neck=neck
        
#         # self.rpn_head=rpn_head

    
#     def forward(self,img,mask,text=None,extra:dict=None):
#         import torch.nn.functional as F
#         """Directly extract features from the backbone+neck."""
#         x = self.backbone(img) #img[bs, C, H, W](8,3,640,640) text[bs, D](8,768)

#         x = self.neck(x)
#         out_mask=[F.interpolate(mask[None].float(), size=_.shape[-2:]).to(torch.bool)[0] for _ in x]
    
#         # flatten 
#         shape=[_.shape[-2:] for _ in x] 
#         x=[rearrange(_,'B C H W -> (H W) B C') for _ in x]
#         out_mask=[rearrange(_,'B H W -> B (H W)') for _ in out_mask]
#         return x,out_mask



# class QRNet_DETR_Enc(nn.Module):
#     def __init__(self, args, QRNet, transformerEnc):
#         super().__init__()
#         hidden_dim=args.vl_hidden_dim
#         divisor = 16 if args.dilation else 32
#         self.QRNet=QRNet
#         self.transformer=transformerEnc
#         self.scale_embed=nn.Embedding(1,self.QRNet.num_channels)
#         self.proj=nn.Linear(self.QRNet.num_channels,hidden_dim)
#         self.visu_pos_embed=nn.Embedding(int(args.imsize/divisor)*int(args.imsize/divisor), hidden_dim)
#     def forward(self, img, mask, text=None):
#         x, out_mask=self.QRNet(img, mask ,text)
#         visu_src=x[-1]
#         N, bs, _ = visu_src.shape
#         visu_mask=out_mask[-1]
#         visu_scale=repeat(self.scale_embed.weight[0], 'D -> L B D', B=bs, L=N)
#         visu_src=self.proj(visu_src+visu_scale)
#         L=self.visu_pos_embed.weight.shape[0]
#         visu_pos=self.visu_pos_embed.weight.reshape(int(sqrt(L)), int(sqrt(L)), -1)
#         visu_pos=visu_pos.unsqueeze(2).repeat(1,1,bs,1).permute(2,3,0,1)
#         out=self.transformer(visu_src.permute(1,2,0), visu_mask, visu_pos)
#         return out[1], visu_mask, visu_pos.flatten(2).permute(2,0,1)

# def build_QRNet(args):
#     qrnet=QRNet(args)
#     transformerEnc=build_transformer(args)
#     qrnet_transEnc=QRNet_DETR_Enc(args, qrnet, transformerEnc)
#     return qrnet_transEnc

if __name__=='__main__':
    pass
        
        

