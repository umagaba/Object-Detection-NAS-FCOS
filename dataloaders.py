import os, json, torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

class COCOFullDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=800):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.img_size = img_size

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(f"{self.img_dir}/{info['file_name']}").convert("RGB")
        
        w, h = img.size
        scale = self.img_size / min(w, h)
        if max(w, h) * scale > 1333: scale = 1333 / max(w, h)
        img = TF.resize(img, (int(round(h * scale)), int(round(w * scale))))
        
        cat_id_to_idx = {cid: i for i, cid in enumerate(sorted(self.coco.getCatIds()))}
        boxes = []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            x, y, w_box, h_box = ann["bbox"]
            if w_box > 2 and h_box > 2:
                boxes.append([x * scale, y * scale, (x + w_box) * scale, (y + h_box) * scale, cat_id_to_idx[ann["category_id"]]])
        
        if not boxes: boxes = [[0., 0., 10., 10., 0.]]
        img = TF.normalize(TF.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img, torch.tensor(boxes, dtype=torch.float32)

def coco_full_collate(batch):
    import torch.nn.functional as F
    imgs, boxes = zip(*batch)
    max_h, max_w = max(img.shape[1] for img in imgs), max(img.shape[2] for img in imgs)
    padded = [F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])) for img in imgs]
    return torch.stack(padded), list(boxes)

class COCOValDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=800):
        self.img_dir = img_dir
        with open(ann_file) as f: data = json.load(f)
        self.imgs = {img["id"]: img for img in data["images"]}
        self.img_ids = list(self.imgs.keys())
        self.idx_to_cat = {i: cid for i, cid in enumerate(sorted({c["id"] for c in data["categories"]}))}
        self.img_size = img_size

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.imgs[img_id]
        img = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        w, h = img.size
        scale = self.img_size / min(w, h)
        if max(w, h) * scale > 1333: scale = 1333 / max(w, h)
        img = TF.resize(img, (int(round(h * scale)), int(round(w * scale))))
        img = TF.normalize(TF.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img, img_id, scale, w, h

def val_collate(batch):
    import torch.nn.functional as F
    imgs, img_ids, scales, orig_ws, orig_hs = zip(*batch)
    max_h, max_w = max(img.shape[1] for img in imgs), max(img.shape[2] for img in imgs)
    padded = [F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])) for img in imgs]
    return torch.stack(padded), list(img_ids), list(scales), list(orig_ws), list(orig_hs)