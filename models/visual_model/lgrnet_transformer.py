import math
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention

class LGRNET_Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, prediction_token_num=3,
                 return_intermediate_dec=False):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, prediction_token_num)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead


    def forward(self, memory, vis_mask, pos_embed, pl, txt_mask, query_embed):
        tgt = memory
        hs = self.decoder(memory=pl, memory_key_padding_mask=txt_mask, pos=query_embed, 
                        tgt=tgt, tgt_key_padding_mask=vis_mask, query_pos=pos_embed)
        return hs

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor):     # [num_query(256), bs, 2]
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    # dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale       # [num_query, bs]
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t         # [num_query, bs, 128]
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos                                  # [num_query, bs, 256]

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(256, 256, 256, 2)
        self.coordinate_head = MLP(256, 256, 2, 2)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,    
                memory_key_padding_mask: Optional[Tensor] = None, 
                pos: Optional[Tensor] = None,                     
                query_pos: Optional[Tensor] = None):
        output = tgt
        coordinate_points = self.coordinate_head(memory[0:1]).sigmoid()
        intermediate=[]
        for layer_id, layer in enumerate(self.layers):                      
            # get sine embedding for the coordinate_points
            coordinate_points_embed = gen_sineembed_for_position(coordinate_points)

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, 
                           coordinate_points_embed=coordinate_points_embed,
                           is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        out=output
        if self.norm is not None:
            out=self.norm(out)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(out)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return out.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, prediction_token_num=3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead
        self.prediction_token_num=prediction_token_num

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,  #[20*8*256]
                     query_pos: Optional[Tensor] = None,
                     coordinate_points_embed: Optional[Tensor] = None,
                     is_first = False):
        
        # BERT[CLS] coord. for self-attention
        q = k = torch.cat([self.with_pos_embed(tgt[0:self.prediction_token_num], coordinate_points_embed), self.with_pos_embed(tgt[self.prediction_token_num:], query_pos[self.prediction_token_num:])], dim=0)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=memory, attn_mask=memory_mask, 
                               key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        pass

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                coordinate_points_embed: Optional[Tensor] = None,
                is_first = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 coordinate_points_embed, is_first)

def build_transformer(args):
    return LGRNET_Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.vl_dec_layers,
        normalize_before=args.pre_norm,
        prediction_token_num=args.prediction_token_num,
        return_intermediate_dec=args.return_intermediate_dec,
    )

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
