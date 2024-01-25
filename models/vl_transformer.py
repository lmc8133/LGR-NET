# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import torch
from torch import nn
from .visual_model.lgrnet_transformer import build_transformer


class VL_decoder(nn.Module):
    def __init__(self, args, transformer):
        super().__init__()
        self.args=args
        self.transformer = transformer
        self.prediction_token_num = args.prediction_token_num
        self.text_query_embed=nn.Embedding(args.max_query_len, args.hidden_dim)

    def forward(self, fv, fl, vis_mask, text_mask, pos_embed):
        N, B, C = fv.shape

        pr = torch.zeros([self.prediction_token_num,B,C]).cuda()
        fv = torch.cat((pr, fv), dim=0)
        pr_mask = torch.zeros((B, self.prediction_token_num)).cuda()    # false mask for origin image part, true for padding part
        vis_mask = torch.cat([pr_mask, vis_mask],dim=-1).bool()

        pl = fl.permute(1, 0, 2)
        query_embed = self.text_query_embed.weight.unsqueeze(1).repeat(1, B, 1)
       
        out = self.transformer(memory=fv, vis_mask=vis_mask, pos_embed=pos_embed, 
                                pl=pl, txt_mask=text_mask, query_embed=query_embed)
        merged_cls = out[:,0:self.prediction_token_num,...].permute(0, 2, 1, 3)

        if merged_cls.shape[2] > 1: # multiple prediction tokens
            merged_cls = torch.mean(merged_cls, dim = 2, keepdim = True)    # mean pooling
        return merged_cls.squeeze(2)

def build_vl_decoder(args):
    transformer=build_transformer(args)
    return VL_decoder(args, transformer)