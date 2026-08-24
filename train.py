import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import math
import sys
import gc
import ctypes
import argparse
import random
import numpy as np
import copy

from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as T
from torchvision.ops import masks_to_boxes
from torchvision import tv_tensors 
import torch.nn.functional as F

# ÂºïÁî® Dataset ??Model
from model.dataset import AmodalTomatoDataset, get_transform
from model.model import AISFormerAmodal

# ?™Â?Ë®≠Â?
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# [?úÈçµ 3] Ë®≠Â? Multiprocessing ?üÂ??áÂÖ±‰∫´Á???if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        # ‰ΩøÁî® spawn Ê®°Â??üÂ?Â≠êÈÄ≤Á? (PyTorch Âª∫Ë≠∞)
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        # [?ëÂëΩÁ®ªË?] ?πÁî® file_system ?øÂ? /dev/shm ?ÜÊªøÂ∞éËá¥ SegFault
        mp.set_sharing_strategy('file_system')
    except RuntimeError:
        pass

def collate_fn(batch):
    return tuple(zip(*batch))

def run_garbage_collection():
    """?ãÂ?Ëß∏Áôº?ÉÂúæ?ûÊî∂?áË??∂È??ãÊîæ"""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==============================================================================
# GPU ?∏Ê?Â¢ûÂº∑ (??Copy-Paste)
# ==============================================================================

class GPUAugmentation(torch.nn.Module):
    def __init__(self, use_copy_paste=False, p=0.5):
        super().__init__()
        self.use_copy_paste = use_copy_paste
        self.p = p
        
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5)
        ])

    def apply_copy_paste(self, images, targets):
        """
        ?∑Ê??ÆÊ??áÈ??∂Á? Amodal Copy-Paste:
        1. ?ê‰?Ê™¢Êü•Ë¶ÅË≤º‰∏äÁ??©‰ª∂??        2. Â¶ÇÊ?Ë≤º‰?ÂæåÊ?Â∞éËá¥?æÊ??©‰ª∂?ÑÂèØË¶ãÈù¢Á©çÊ??∞Â??¨Á? 25% ‰ª•‰?ÔºåÂ??®Ê?Ë©≤Ë≤º‰∏äÁâ©‰ª∂Ô?‰øùË≠∑?æÊ??©‰ª∂?ØËæ®Ë≠òÂ∫¶Ôºâ„Ä?        """
        batch_size = len(images)
        if batch_size < 2: return images, targets 
        new_images, new_targets = [], []
        
        for i in range(batch_size):
            dst_img = images[i].clone() 
            dst_target = copy.deepcopy(targets[i])
            
            if random.random() < self.p:
                src_idx = random.choice([idx for idx in range(batch_size) if idx != i])
                src_img, src_target = images[src_idx], targets[src_idx]
                
                if len(src_target["gt_amodal_masks"]) > 0:
                    dst_h, dst_w = dst_img.shape[-2:]
                    src_h, src_w = src_img.shape[-2:]
                    
                    src_img_p, src_amo_p = src_img, src_target["gt_amodal_masks"]
                    if (src_h, src_w) != (dst_h, dst_w):
                        src_img_p = F.interpolate(src_img.unsqueeze(0), size=(dst_h, dst_w), mode="bilinear", align_corners=False).squeeze(0)
                        src_amo_p = F.interpolate(src_amo_p.unsqueeze(1).float(), size=(dst_h, dst_w), mode="nearest").squeeze(1).byte()

                    # Ê∫ñÂ??ê‰??ëÈÅ∏‰∏¶Ê™¢??                    num_src = len(src_amo_p)
                    k = random.randint(1, min(num_src, 3))
                    indices = torch.randperm(num_src, device=dst_img.device)[:k]
                    
                    final_pasted_amos = []
                    final_pasted_vis = []
                    final_pasted_labels = []

                    for idx in indices:
                        proposed_amo = src_amo_p[idx]
                        proposed_label = src_target["labels"][idx]
                        
                        # --- ?ÆÊ??áÊ™¢??(‰øùË≠∑Ê©üÂà∂) ---
                        is_safe = True
                        
                        # 1. Ê™¢Êü•Â∞ç„ÄåÂ??ñÂ∑≤?âÁâ©‰ª∂„ÄçÁ?ÂΩ±Èüø
                        if len(dst_target["gt_visible_masks"]) > 0:
                            old_areas = dst_target["gt_visible_masks"].flatten(1).sum(dim=1)
                            new_areas = (dst_target["gt_visible_masks"] & (~proposed_amo)).flatten(1).sum(dim=1)
                            
                            m = old_areas > 0
                            if m.any():
                                ratios = new_areas[m].float() / old_areas[m].float()
                                if (ratios < 0.25).any(): is_safe = False
                        
                        # 2. Ê™¢Êü•Â∞ç„ÄåÊú¨Ê¨°Â∑≤Ë≤º‰??©‰ª∂?çÁ?ÂΩ±Èüø
                        if is_safe and len(final_pasted_vis) > 0:
                            p_vis_stack = torch.stack(final_pasted_vis)
                            old_p_areas = p_vis_stack.flatten(1).sum(dim=1)
                            new_p_areas = (p_vis_stack & (~proposed_amo)).flatten(1).sum(dim=1)
                            
                            m_p = old_p_areas > 0
                            if m_p.any():
                                p_ratios = new_p_areas[m_p].float() / old_p_areas[m_p].float()
                                if (p_ratios < 0.25).any(): is_safe = False
                        
                        # --- ?∑Ë?Á≤òË≤º (?´È?Á∑?π≥Êª?Edge Blending) ---
                        if is_safe:
                            # 1. ?¢Á?Âπ≥Ê???Alpha Mask (‰ΩøÁî®?üÊ≠£??Gaussian Blur Ê∂àÈô§?üÁ°¨?äÁ∑£?èÁ??∑Â±§)
                            proposed_amo_float = proposed_amo.float().unsqueeze(0) # [1, H, W]
                            # ‰ΩøÁî® sigma=2.0 ?≤Ë??îÂ??äÁ∑£ÔºåÈÅø?çÈ??ªÁâπÂæµÂ???loss_objectness ?áÁõ™
                            import torchvision.transforms.functional as TF
                            alpha_soft = TF.gaussian_blur(proposed_amo_float, kernel_size=[5, 5], sigma=[2.0, 2.0])
                            alpha_soft = alpha_soft.squeeze(0) # [H, W]
                            
                            # 2. ?∑Ë? Alpha Blending
                            # dst_img: [3, H, W], src_img_p: [3, H, W]
                            dst_img = dst_img * (1.0 - alpha_soft) + src_img_p * alpha_soft
                            
                            # ?¥Êñ∞?æÊ??©‰ª∂?ÆÁΩ©
                            dst_target["gt_visible_masks"] = dst_target["gt_visible_masks"] & (~proposed_amo)
                            if "gt_background_objs_masks" in dst_target:
                                dst_target["gt_background_objs_masks"] = dst_target["gt_background_objs_masks"] & (~proposed_amo)
                            
                            # ?¥Êñ∞?¨Ê¨°Â∑≤Ë≤º‰∏äÁâ©‰ª∂Á??ØË??Ä??(Ë¢´Â?‰æÜËÄÖÈÅÆ??
                            for j in range(len(final_pasted_vis)):
                                final_pasted_vis[j] = final_pasted_vis[j] & (~proposed_amo)
                                
                            final_pasted_amos.append(proposed_amo)
                            final_pasted_vis.append(proposed_amo) # ?õË≤º‰∏äÊ??ØË? = ?®Ê®°??                            final_pasted_labels.append(proposed_label)

                    # ?à‰ΩµÁµêÊ?
                    if len(final_pasted_labels) > 0:
                        p_vis = torch.stack(final_pasted_vis)
                        p_amo = torch.stack(final_pasted_amos)
                        p_lab = torch.stack(final_pasted_labels)
                        
                        dst_target["gt_visible_masks"] = torch.cat([dst_target["gt_visible_masks"], p_vis], dim=0)
                        dst_target["gt_amodal_masks"] = torch.cat([dst_target["gt_amodal_masks"], p_amo], dim=0)
                        dst_target["labels"] = torch.cat([dst_target["labels"], p_lab], dim=0)
                        
                        if "gt_background_objs_masks" in dst_target:
                            bg_fill = torch.zeros((len(p_lab), dst_h, dst_w), dtype=torch.uint8, device=dst_img.device)
                            dst_target["gt_background_objs_masks"] = torch.cat([dst_target["gt_background_objs_masks"], bg_fill], dim=0)
                    
                    # ?çÊñ∞?åÊ≠•?éÊøæ?áÊõ¥??BBox
                    mask_areas = dst_target["gt_visible_masks"].flatten(1).sum(dim=1)
                    keep = mask_areas > 1
                    if keep.any():
                        for k in ["labels", "gt_visible_masks", "gt_amodal_masks", "gt_background_objs_masks"]:
                            if k in dst_target and dst_target[k] is not None: dst_target[k] = dst_target[k][keep]
                        dst_target["boxes"] = masks_to_boxes(dst_target["gt_visible_masks"])
                        dst_target["iscrowd"] = torch.zeros(len(dst_target["labels"]), dtype=torch.int64, device=dst_img.device)
                        dst_target["area"] = (dst_target["boxes"][:, 2] - dst_target["boxes"][:, 0]) * (dst_target["boxes"][:, 3] - dst_target["boxes"][:, 1])
                    else:
                        dst_target["boxes"] = torch.empty((0, 4), device=dst_img.device)
                        dst_target["labels"] = torch.empty((0,), dtype=torch.int64, device=dst_img.device)
                        dst_target["gt_visible_masks"] = torch.empty((0, dst_h, dst_w), dtype=torch.uint8, device=dst_img.device)

            new_images.append(dst_img)
            new_targets.append(dst_target)
        return new_images, new_targets

    def forward(self, images, targets):
        if self.use_copy_paste:
            images, targets = self.apply_copy_paste(images, targets)

        final_images, final_targets = [], []
        for img, target in zip(images, targets):
            h, w = img.shape[-2:]
            img_tv = tv_tensors.Image(img)
            try:
                target_tv = {
                    "boxes": tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=(h, w)),
                    "labels": target["labels"],
                    "gt_visible_masks": tv_tensors.Mask(target["gt_visible_masks"]),
                    "gt_amodal_masks": tv_tensors.Mask(target["gt_amodal_masks"]),
                }
                if "gt_background_objs_masks" in target: target_tv["gt_background_objs_masks"] = tv_tensors.Mask(target["gt_background_objs_masks"])
                img_aug, target_aug = self.transforms(img_tv, target_tv)
            except Exception: img_aug, target_aug = img_tv, target
            
            if "gt_visible_masks" in target_aug and target_aug["gt_visible_masks"].shape[0] > 0:
                mask_areas = target_aug["gt_visible_masks"].flatten(1).sum(dim=1)
                keep = mask_areas > 1
                if keep.any():
                    for k in ["labels", "gt_visible_masks", "gt_amodal_masks", "gt_background_objs_masks", "boxes"]:
                        if k in target_aug and target_aug[k] is not None: target_aug[k] = target_aug[k][keep]
                    target_aug["boxes"] = masks_to_boxes(target_aug["gt_visible_masks"])
                    b = target_aug["boxes"]
                    target_aug["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                    target_aug["iscrowd"] = torch.zeros(len(b), dtype=torch.int64, device=img.device)
                else:
                    target_aug["boxes"] = torch.empty((0, 4), device=img.device)
                    for k in ["labels", "gt_visible_masks", "gt_amodal_masks", "gt_background_objs_masks"]:
                        if k in target_aug: target_aug[k] = target_aug[k][:0]
            
            target_aug["height"], target_aug["width"] = torch.tensor(img_aug.shape[-2], device=img.device), torch.tensor(img_aug.shape[-1], device=img.device)
            final_images.append(img_aug.as_subclass(torch.Tensor))
            for k in ["boxes", "gt_visible_masks", "gt_amodal_masks", "gt_background_objs_masks"]:
                if k in target_aug: target_aug[k] = target_aug[k].as_subclass(torch.Tensor)
            final_targets.append(target_aug)
        return final_images, final_targets

@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.train() 
    total_losses = {} 
    steps = 0
    print("\n?? Starting Validation...")
    for i, (images, targets) in enumerate(data_loader):
        if i >= 50: break 
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            loss_dict = model(images, targets)
        for k, v in loss_dict.items(): total_losses[k] = total_losses.get(k, 0.0) + v.item()
        total_losses["total_loss"] = total_losses.get("total_loss", 0.0) + sum(loss for loss in loss_dict.values()).item()
        steps += 1
    return {k: v / max(steps, 1) for k, v in total_losses.items()}

def main(args):
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "epochs": 100000, "batch_size": args.batch_size, "learning_rate": args.lr,
        "gradient_accumulation_steps": args.grad_accum, "output_dir": args.output_dir, "log_tensorboard_dir": "logs",
        "save_interval": 1000, "max_iter": 50000, "warmup_iters": 1000,
        "lr_scheduler_gamma": 0.1,
        "save_freq": 1000, "val_freq": 500, "print_freq": 20,
        "num_workers": 4, "prefetch_factor": 2, "persistent_workers": False, "resume_path": args.resume, 
        "backbone": args.backbone, "use_copy_paste": args.use_copy_paste,
        "use_light_mask_head": True, "official_loss_weight": args.official_loss_weight
    }
    if not args.resume: config["output_dir"] = os.path.join(config["output_dir"], f"{config['backbone']}_DA_Run")
    os.makedirs(config["output_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(config["output_dir"], config["log_tensorboard_dir"]))
    from torch.utils.data import ConcatDataset
    train_dataset = AmodalTomatoDataset('./datasets/KINS/instances_train.json', './datasets/KINS/training/image_2', transforms=get_transform(train=True), target_size=3000)
    val_dataset = AmodalTomatoDataset('./datasets/KINS/instances_val.json', './datasets/KINS/testing/image_2', transforms=get_transform(train=False), target_size=3000)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], collate_fn=collate_fn, pin_memory=True, prefetch_factor=config["prefetch_factor"], persistent_workers=config["persistent_workers"])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    model = AISFormerAmodal(
        num_classes=9, 
        min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), 
        max_size=3000, 
        backbone_name=config['backbone'], 
        bbox_reg_weights=(10.0, 10.0, 5.0, 5.0), 
        frozen_stages=0,
        use_light_mask_head=config.get('use_light_mask_head', False),
        official_loss_weight=config.get('official_loss_weight', False),
        n_heads=2,
        dim_feedforward=2048
    )
    model.to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_iter"] - config["warmup_iters"], eta_min=1e-6)
    scaler = GradScaler(); global_step = 0; best_val_loss = float('inf')
    if config["resume_path"] and os.path.exists(config["resume_path"]):
        checkpoint = torch.load(config['resume_path'], map_location=config["device"])
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        global_step = checkpoint.get("global_step", 0); best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    gpu_augmentor = GPUAugmentation(use_copy_paste=config["use_copy_paste"]).to(config["device"])
    print(f"?é¨ Start training loop (Copy-Paste: {config['use_copy_paste']})...")
    model.train(); data_loader_iter = iter(train_loader)
    try:
        while global_step < config["max_iter"]:
            try: images, targets = next(data_loader_iter)
            except StopIteration: data_loader_iter = iter(train_loader); images, targets = next(data_loader_iter)
            images = [img.to(config["device"], non_blocking=True) for img in images]
            targets = [{k: v.to(config["device"], non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            with torch.no_grad(): images, targets = gpu_augmentor(images, targets)
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            if not math.isfinite(loss_value):
                print(f"?†Ô? Loss is {loss_value}, skipping step."); optimizer.zero_grad(); run_garbage_collection(); continue
            scaler.scale(losses).backward(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            if global_step < config["warmup_iters"]:
                cur_lr = config["learning_rate"] * (0.001 + (1.0 - 0.001) * float(global_step) / float(config["warmup_iters"]))
                for pg in optimizer.param_groups: pg['lr'] = cur_lr
            else: lr_scheduler.step()
            global_step += 1
            if global_step % config["print_freq"] == 0:
                print(f"Iter: {global_step}/{config['max_iter']} | Loss: {loss_value:.4f}")
                writer.add_scalar('Train/Total_Loss', loss_value, global_step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f'Train/{k}', v.item(), global_step)

            if global_step % config["val_freq"] == 0:
                val_losses = evaluate_loss(model, val_loader, config["device"])
                current_val_loss = val_losses['total_loss']
                writer.add_scalar('Validation/Total_Loss', current_val_loss, global_step)
                for k, v in val_losses.items():
                    if k != "total_loss":
                        writer.add_scalar(f'Validation/{k}', v, global_step)
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save({"model_state_dict": model.state_dict(), "global_step": global_step, "best_val_loss": best_val_loss}, os.path.join(config["output_dir"], "model_best.pth"))
                model.train(); run_garbage_collection()
            if global_step % config["save_freq"] == 0:
                torch.save({"model_state_dict": model.state_dict(), "global_step": global_step, "best_val_loss": best_val_loss}, os.path.join(config["output_dir"], f"checkpoint_iter_{global_step}.pth"))
    except Exception as e:
        import traceback; traceback.print_exc()
    finally: writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--output_dir", type=str, default="output/kins_2026/run_official", help="Directory to save logs and weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--use_copy_paste", action="store_true", help="Enable copy-paste augmentation")
    parser.add_argument("--batch_size", type=int, default=1, help="Physical batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--official_loss_weight", action="store_true", help="Scale occluder and invisible mask losses by 0.25 as in the official repo")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 for AdamW)")
    args = parser.parse_args()
    main(args)
