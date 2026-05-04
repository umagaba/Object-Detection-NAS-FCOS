import os, json, csv, torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from models import NASFCOSDetector
from controller import NASController, reinforce_update
from dataloaders import COCOFullDataset, COCOValDataset, coco_full_collate, val_collate
from utils import compute_fcos_loss, decode_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "./outputs"
DATA_DIR = "./data"
os.makedirs(OUT_DIR, exist_ok=True)

def download_datasets():
    print("Downloading datasets via HuggingFace...")
    path = snapshot_download(repo_id="rafaelpadilla/coco2017", repo_type="dataset", local_dir=DATA_DIR)
    return {
        "train_img": os.path.join(path, "train2017"),
        "train_ann": os.path.join(path, "annotations/instances_train2017.json"),
        "val_img": os.path.join(path, "val2017"),
        "val_ann": os.path.join(path, "annotations/instances_val2017.json"),
    }

def phase1_search_smoke_test(paths):
    print("\n--- Phase 1: Controller Search (SMOKE TEST: 1 Arch, 10 Steps) ---")
    controller = NASController().to(DEVICE)
    
    # 1. Sample EXACTLY 1 architecture
    controller.eval()
    best_fpn, _, _ = controller.sample_fpn_arch()
    best_head, best_share, _, _ = controller.sample_head_arch()
    
    print(f"Sampled FPN Blocks: {len(best_fpn)}")
    print(f"Sampled Head: {best_head}")
    
    # 2. Test the architecture syntax (Forward/Backward pass for 10 steps)
    model = NASFCOSDetector(best_fpn, best_head, share_from=best_share).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=8e-4)
    
    ds = COCOFullDataset(paths["train_img"], paths["train_ann"])
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=coco_full_collate)
    
    model.train()
    print("Testing proxy architecture forward/backward passes...")
    for step, (imgs, boxes) in enumerate(loader):
        if step >= 10: # HARD LIMIT: 10 steps
            break 
            
        imgs = imgs.to(DEVICE)
        boxes = [b.to(DEVICE) for b in boxes]
        
        opt.zero_grad()
        cls_p, reg_p, ctr_p = model(imgs)
        loss, _ = compute_fcos_loss(cls_p, reg_p, ctr_p, boxes, 800, 80) # Using 80 classes for COCO
        loss.backward()
        opt.step()
        print(f"  Proxy Step {step+1}/10 - Loss: {loss.item():.4f}")

    # 3. Save Search results
    res = {"best_fpn_arch": best_fpn, "best_head_arch": best_head, "best_share_from": best_share}
    with open(f"{OUT_DIR}/nas_fcos_search_results.json", "w") as f: 
        json.dump(res, f, indent=2)
        
    return res

def full_training_smoke_test(config, paths):
    print("\n--- Phase 2: Full Architecture Training (SMOKE TEST: 10 Steps) ---")
    model = NASFCOSDetector(config["best_fpn_arch"], config["best_head_arch"], share_from=config["best_share_from"]).to(DEVICE)
    
    ds = COCOFullDataset(paths["train_img"], paths["train_ann"])
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=coco_full_collate)
    
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    with open(f"{OUT_DIR}/nas_fcos_training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        
        model.train()
        for step, (imgs, boxes) in enumerate(loader):
            if step >= 10: # HARD LIMIT: 10 steps
                break 
                
            imgs = imgs.to(DEVICE)
            boxes = [b.to(DEVICE) for b in boxes]
            
            opt.zero_grad()
            cls_p, reg_p, ctr_p = model(imgs)
            loss, _ = compute_fcos_loss(cls_p, reg_p, ctr_p, boxes, 800, 80)
            loss.backward()
            opt.step()
            
            writer.writerow([step, loss.item()])
            print(f"  Train Step {step+1}/10 - Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), f"{OUT_DIR}/nas_fcos_final.pth")
    return model

def evaluation_smoke_test(model, paths):
    print("\n--- Phase 3: Evaluation (SMOKE TEST: 5 Batches) ---")
    ds = COCOValDataset(paths["val_img"], paths["val_ann"])
    # Batch size 2 * 5 steps = 10 images tested
    loader = DataLoader(ds, batch_size=2, collate_fn=val_collate) 
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for step, (imgs, img_ids, scales, _, _) in enumerate(loader):
            if step >= 5: # HARD LIMIT: 5 batches
                break
                
            imgs = imgs.to(DEVICE)
            cls_p, reg_p, ctr_p = model(imgs)
            
            for i in range(imgs.shape[0]):
                boxes, scores, labels = decode_predictions(
                    [c[i:i+1] for c in cls_p], 
                    [r[i:i+1] for r in reg_p], 
                    [ct[i:i+1] for ct in ctr_p], 
                    800
                )
                if len(boxes) == 0: continue
                
                boxes = (boxes.cpu() / scales[i]).tolist()
                scores, labels = scores.cpu().tolist(), labels.cpu().tolist()
                
                for b, s, l in zip(boxes, scores, labels):
                    results.append({
                        "image_id": img_ids[i], 
                        "category_id": ds.idx_to_cat[l], 
                        "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], 
                        "score": s
                    })
            print(f"  Eval Batch {step+1}/5 processed.")

    res_file = f"{OUT_DIR}/nas_fcos_val_results.json"
    with open(res_file, "w") as f: 
        json.dump(results, f)
    print(f"Evaluation complete. Found {len(results)} detections.")

if __name__ == "__main__":
    paths = download_datasets()
    config = phase1_search_smoke_test(paths)
    model = full_training_smoke_test(config, paths)
    evaluation_smoke_test(model, paths)
