import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import is_dist_avail_and_initialized, get_world_size
from .language_model.bert import build_bert
from .vl_transformer import build_vl_decoder
from utils import box_utils
from utils.box_utils import generalized_box_iou
import numpy as np

class MuModule:
    pass

class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32

        from models.QRNet import QRNet
        self.visumodel = QRNet(args)

        visu_seq_len = int((args.imsize/divisor)**2+(args.imsize/(2*divisor))**2)
        self.visu_pos_embed=nn.Embedding(visu_seq_len+args.prediction_token_num, hidden_dim)

        self.textmodel = build_bert(args)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)#256*256
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)#768*256

        self.vl_transformer_decoder = build_vl_decoder(args)
        
        self.prediction = MLP(hidden_dim, hidden_dim, 4, 3)
        self.class_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bertcls_proj = nn.Linear(768, hidden_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        for p in self.prediction.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        device = img_data.tensors.device

        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()  #[bs, N, C] [bs, N]
        
        assert text_mask is not None
        text_cls=text_src[:, 0]
        text_src = self.text_proj(text_src) #[bs, N, c]

        # visual backbone
        x, out_mask=self.visumodel(img_data.tensors, img_data.mask, text_cls)
        visu_mask=torch.cat(out_mask[-2:],dim=1)
        visu_src=torch.cat(x[-2:],dim=0)
        
        visu_src = self.visu_proj(visu_src) # (N*B)xC

        visu_pos=self.visu_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vg_hs = self.vl_transformer_decoder(visu_src, text_src, visu_mask, text_mask, visu_pos)

        pred_class = F.normalize(self.class_proj(vg_hs[-1]), dim = -1)
        bertcls_class = F.normalize(self.bertcls_proj(text_cls), dim = -1)
        aux_boxes = self.prediction(vg_hs).sigmoid()
        pred_box={'pred_boxes':aux_boxes[-1]}
        if self.training:
            pred_box['aux_boxes']=[{'pred_boxes':b} for b in aux_boxes[:-1]]

        crossmodal_loss = bs - F.cosine_similarity(bertcls_class, pred_class, dim=-1).sum()
        
        sim_i2t = pred_class @ bertcls_class.T * self.logit_scale.exp()
        sim_t2i = sim_i2t.t()
        labels = torch.arange(bs, device = device, dtype = torch.long)
        crossmodal_loss = (F.cross_entropy(sim_i2t, labels) +
                      F.cross_entropy(sim_t2i, labels)) / 2
        
        return pred_box, crossmodal_loss

def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class VGCriterion(nn.Module):
    """ This class computes the loss for VLTVG."""
    def __init__(self, weight_dict):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict

    def loss_boxes(self, outputs, target_boxes, num_pos):
        """Compute the losses related to the bounding boxes (the L1 regression loss and the GIoU loss)"""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        src_boxes = box_utils.xywh2xyxy(src_boxes)
        target_boxes = box_utils.xywh2xyxy(target_boxes)
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses['giou'] = loss_giou.sum() / num_pos
        return losses


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """
        gt_boxes = targets
        pred_boxes = outputs['pred_boxes']

        losses = {}
        B, _ = pred_boxes.shape
        num_pos = avg_across_gpus(pred_boxes.new_tensor(B))
        loss = self.loss_boxes(outputs, gt_boxes, num_pos)
        losses.update(loss)

        # Apply the loss function to the outputs from all the stages
        if 'aux_boxes' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_boxes']):
                l_dict = self.loss_boxes(aux_outputs, gt_boxes, num_pos)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

def build_vgmodel(args):
    device = torch.device(args.device)

    model = TransVG(args=args)

    weight_dict = {'loss_cls': 1, 'l1': args.bbox_loss_coef}
    weight_dict['giou'] = args.giou_loss_coef
    weight_dict.update(args.other_loss_coefs)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.vl_dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = VGCriterion(weight_dict=weight_dict)
    criterion.to(device)

    return model, criterion