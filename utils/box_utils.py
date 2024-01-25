import torch
import math
from torchvision.ops.boxes import box_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def xywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def xyxy2xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(pred, target):
    area1 = box_area(pred)
    area2 = box_area(target)

    lt = torch.max(pred[:, None, :2], target[:, :2])  # [N,M,2]
    rb = torch.min(pred[:, None, 2:], target[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(pred, target):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(pred)
    and M = len(target)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (pred[:, 2:] >= pred[:, :2]).all()
    assert (target[:, 2:] >= target[:, :2]).all()
    iou, union = box_iou(pred, target)

    lt = torch.min(pred[:, None, :2], target[:, :2])
    rb = torch.max(pred[:, None, 2:], target[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def distance_box_iou(pred, target):           #lmc_added
    """
    Complete IoU from https://ojs.aaai.org/index.php/AAAI/article/view/6999

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(pred)
    and M = len(target)
    """
    assert (pred[:, 2:] >= pred[:, :2]).all()
    assert (target[:, 2:] >= target[:, :2]).all()
    iou, uniou=box_iou(pred, target)
    
    rows=pred.shape[0]
    cols=target.shape[0]
    dious=torch.zeros((rows, cols))
    
    # w1=pred[:, 2]-pred[:, 0]
    # h1=pred[:, 3]-pred[:, 1]
    # w2=target[:, 2]-target[:, 0]
    # h2=target[:, 3]-target[:, 1]

    center_x1=(pred[:, 2]+pred[:, 0])/2
    center_y1=(pred[:, 3]+pred[:, 1])/2
    center_x2=(target[:, 2]+target[:, 0])/2
    center_y2=(target[:, 3]+target[:, 1])/2

    out_max_xy=torch.max(pred[:, 2:], target[:, 2:])
    out_min_xy=torch.min(pred[:, :2], target[:, :2])
    
    inter_diag=(center_x2-center_x1)**2+(center_y2-center_y1)**2
    outer=torch.clamp((out_max_xy-out_min_xy), min=0)
    outer_diag=(outer[:, 0]**2)+(outer[:, 1]**2)
    u=inter_diag/outer_diag
    # with torch.no_grad():
    #     arctan=torch.atan(w2/h2)-torch.atan(w1/h1)
    #     v=(4/(math.pi**2))*torch.pow((torch.atan(w2/h2)-torch.atan(w1/h1)), 2)
    #     S=1-iou
    #     alpha=v/(S+v)
    #     w_tmp=2*w1
    # ar=(8/(math.pi**2))*arctan*((w1-w_tmp)*h1)
    # if CIou:
    #     cious=iou-(u+alpha*ar)
    # else:
    dious=iou-u         #DIoU
    dious=torch.clamp(dious, min=-1.0, max=1.0)
    return dious

def complete_box_iou(pred, target, eps=1e-7):              #lmc add from mmdetection
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    dious=distance_box_iou(pred, target)

    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v)           #不懂为啥这么实现

    # CIoU
    cious = ious - (rho2 / c2 + alpha * v)
    loss = cious.clamp(min=-1.0, max=1.0)
    return loss    