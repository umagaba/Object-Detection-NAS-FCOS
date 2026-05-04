import torch
import torch.nn.functional as F
from torchvision.ops import nms

FPN_STRIDES = [8, 16, 32, 64, 128]

def generate_targets(feature, gt_boxes, image_size, num_classes):
    B, _, H, W = feature.shape
    dev = feature.device
    t_cls = torch.zeros(B, num_classes, H, W, device=dev)
    t_reg = torch.zeros(B, 4, H, W, device=dev)
    t_ctr = torch.zeros(B, 1, H, W, device=dev)

    gy, gx = torch.meshgrid(
        (torch.arange(H, device=dev).float() + 0.5) * (image_size / H),
        (torch.arange(W, device=dev).float() + 0.5) * (image_size / W),
        indexing="ij"
    )

    for b in range(B):
        for box in gt_boxes[b]:
            x1, y1, x2, y2 = box[:4].tolist()
            cls_idx = min(int(box[4]) if box.shape[0] == 5 else 0, num_classes - 1)
            if x2 <= x1 + 1 or y2 <= y1 + 1: continue

            inside = (gx > x1) & (gx < x2) & (gy > y1) & (gy < y2)
            if not inside.any(): continue

            l, t, r, bot = gx - x1, gy - y1, x2 - gx, y2 - gy
            t_cls[b, cls_idx][inside] = 1.0
            t_reg[b, 0][inside], t_reg[b, 1][inside] = l[inside], t[inside]
            t_reg[b, 2][inside], t_reg[b, 3][inside] = r[inside], bot[inside]

            t_ctr[b, 0][inside] = torch.sqrt((torch.min(l, r) / torch.max(l, r).clamp(1e-6)) * (torch.min(t, bot) / torch.max(t, bot).clamp(1e-6)))[inside]
    return t_cls, t_reg, t_ctr

def compute_fcos_loss(cls_preds, reg_preds, ctr_preds, gt_boxes, image_size, num_classes):
    total_cls = total_reg = total_ctr = 0.0
    for lvl, (cp, rp, ctp) in enumerate(zip(cls_preds, reg_preds, ctr_preds)):
        tc, tr, tct = generate_targets(cp, gt_boxes, image_size, num_classes)
        tr = tr / FPN_STRIDES[lvl]
        pos = (tc.sum(dim=1, keepdim=True) > 0).float()
        npos = pos.sum().clamp(min=1.)

        bce = F.binary_cross_entropy_with_logits(cp, tc, reduction="none")
        p_t = torch.sigmoid(cp) * tc + (1 - torch.sigmoid(cp)) * (1 - tc)
        total_cls += ((0.25 * tc + 0.75 * (1 - tc)) * (1 - p_t) ** 2.0 * bce).mean()
        total_reg += (torch.abs(rp - tr) * pos).sum() / npos
        total_ctr += (F.binary_cross_entropy_with_logits(ctp, tct, reduction="none") * pos).sum() / npos

    n = len(cls_preds)
    lc, lr, lct = total_cls/n, total_reg/n, total_ctr/n
    return lc + lr + lct, (lc.item(), lr.item(), lct.item())

def decode_predictions(cls_preds, reg_preds, ctr_preds, image_size, score_thresh=0.05, nms_thresh=0.5, max_dets=100):
    all_boxes, all_scores, all_labels = [], [], []
    for lvl, (cp, rp, ctp) in enumerate(zip(cls_preds, reg_preds, ctr_preds)):
        stride = FPN_STRIDES[lvl]
        scores = (torch.sigmoid(cp[0]) * torch.sigmoid(ctp[0])).sqrt()
        max_scores, labels = scores.max(dim=0)
        keep = max_scores > score_thresh
        if not keep.any(): continue

        gy, gx = torch.meshgrid((torch.arange(cp.shape[2], device=cp.device).float() + 0.5) * stride,
                                (torch.arange(cp.shape[3], device=cp.device).float() + 0.5) * stride, indexing="ij")
        
        x1, y1 = (gx - rp[0, 0] * stride)[keep], (gy - rp[0, 1] * stride)[keep]
        x2, y2 = (gx + rp[0, 2] * stride)[keep], (gy + rp[0, 3] * stride)[keep]

        all_boxes.append(torch.stack([x1, y1, x2, y2], dim=1))
        all_scores.append(max_scores[keep])
        all_labels.append(labels[keep])

    if not all_boxes: return torch.zeros((0,4)), torch.zeros((0,)), torch.zeros((0,))
    boxes, scores, labels = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_size)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_size)

    keep_all = []
    for cls_id in labels.unique():
        m = labels == cls_id
        keep_all.append(m.nonzero(as_tuple=True)[0][nms(boxes[m], scores[m], nms_thresh)])
    
    if not keep_all: return torch.zeros((0,4)), torch.zeros((0,)), torch.zeros((0,))
    keep = torch.cat(keep_all)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(scores) > max_dets:
        topk = scores.topk(max_dets).indices
        boxes, scores, labels = boxes[topk], scores[topk], labels[topk]
    return boxes, scores, labels