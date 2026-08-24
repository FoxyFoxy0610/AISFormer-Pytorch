# ==============================================================================
# Functional Block: Inference and Visualization Block
# Description: This module is responsible for the essential operations of Inference and Visualization Block.
# ==============================================================================
import torch
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
from torchvision.ops import batched_nms
from tqdm import tqdm
from model.model import AISFormerAmodal

# ÂÆöÁæ©È°ûÂà•È°èËâ≤ (Á¥¢Â? 0 ?∫Ë??ØÔ??ôË£°Â∞çÊ? 1~4)
CLASSES = ['BG', 'flower', 'Green', 'Turning', 'Harvest']
COLORS = [
    [0, 0, 0],       # BG
    [0, 255, 255],   # flower (Yellow)
    [0, 255, 0],     # Green (Green)
    [0, 165, 255],   # Turning (Orange)
    [0, 0, 255],     # Harvest (Red)
]

def get_demo_transform():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

def resize_image_aspect_ratio(image, max_size=1280):
    w, h = image.size
    scale = min(max_size / w, max_size / h)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.BILINEAR), scale
    return image, 1.0

def overlay_mask(image, mask, color, alpha=0.5, threshold=0.5):
    """Áπ™Ë£Ω?äÈÄèÊ?Â°´Â? Mask"""
    mask_bool = mask > threshold
    if not mask_bool.any(): return image
    
    roi = image[mask_bool]
    blended = roi * (1 - alpha) + np.array(color) * alpha
    image[mask_bool] = blended.astype(np.uint8)
    return image

def draw_contours(image, mask, color, thickness=2, threshold=0.5):
    """Áπ™Ë£Ω Mask Ëº™Â?"""
    mask_uint8 = (mask > threshold).astype(np.uint8)
    if mask_uint8.max() == 0: return image
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)
    return image

def run_demo_folder(input_folder, output_folder, checkpoint_path, backbone_name='resnet50', conf_thresh=0.3, mask_thresh=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. ËºâÂÖ•Ê®°Â?
    print(f"Loading model from {checkpoint_path}...")
    print(f"Using Backbone: {backbone_name}")

    model = AISFormerAmodal(
        num_classes=5,
        n_heads=4,
        dim_feedforward=1024,
        mask_pooler_resolution=14,
        min_size=800,
        max_size=1333,
        bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
        backbone_name=backbone_name 
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        return

    # Âª∫Á?Ëº∏Âá∫?ÆÈ?ÁµêÊ?
    vis_output_dir = os.path.join(output_folder, "visible")
    amo_output_dir = os.path.join(output_folder, "amodal")
    inv_output_dir = os.path.join(output_folder, "invisible")
    os.makedirs(vis_output_dir, exist_ok=True)
    os.makedirs(amo_output_dir, exist_ok=True)
    os.makedirs(inv_output_dir, exist_ok=True)

    # ?úÂ??Ä?âÂ???    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(image_files)} images in {input_folder}")

    transform = get_demo_transform()

    # ?ãÂ??πÊ¨°?ïÁ?
    for img_name in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_folder, img_name)
        
        # 2. ËÆÄ?ñË??çË???        pil_img = Image.open(image_path).convert("RGB")
        
        # [Ê≠•È? B] Resize (‰ΩøÁî®Â¢ûÂº∑ÂæåÁ???
        pil_img_resized, scale_factor = resize_image_aspect_ratio(pil_img, max_size=1280)
        
        img_tensor = transform(pil_img_resized).to(device)

        # 3. ?®Ë?
        with torch.no_grad():
            predictions = model([img_tensor])[0]

        # 4. NMS ÂæåË???        pred_boxes = predictions['boxes']
        pred_scores = predictions['scores']
        pred_labels = predictions['labels']
        
        keep_indices = batched_nms(pred_boxes, pred_scores, pred_labels, 0.25)
        
        scores = pred_scores[keep_indices].cpu().numpy()
        labels = pred_labels[keep_indices].cpu().numpy()
        boxes = pred_boxes[keep_indices].cpu().numpy()
        
        # ?êÂ? Mask (‰øÆÊ≠£Ôºöpred_visible_masks ?∫ÂèØË¶ãÔ?pred_amodal_masks ??Amodal)
        if 'pred_visible_masks' in predictions:
            vis_masks = predictions['pred_visible_masks'][keep_indices].squeeze(1).cpu().numpy()
        else: vis_masks = None

        if 'pred_amodal_masks' in predictions:
            amo_masks = predictions['pred_amodal_masks'][keep_indices].squeeze(1).cpu().numpy()
        else: amo_masks = None

        if 'pred_invisible_masks' in predictions:
            inv_masks = predictions['pred_invisible_masks'][keep_indices].squeeze(1).cpu().numpy()
        else: inv_masks = None

        # 5. Áπ™Â?
        # ?ôË£°?ëÂÄë‰Ωø?®„ÄåÂ?Âº∑Â??çÁ??ñ‰?Áπ™Â?ÔºåÈÄôÊ®£?®ÂèØ‰ª•Á???CLAHE ?ÑÊ???        # Â¶ÇÊ??≥Áï´?®Â??ñ‰?ÔºåË??πÁî® resize_image_aspect_ratio(pil_img, ...) ?ÑÁ???        img_np = np.array(pil_img_resized)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Âª∫Á?‰∏âÂºµ?®Á??´Â?
        img_vis = img_np.copy()
        img_amo = img_np.copy()
        img_inv = img_np.copy()
        
        for i in range(len(scores)):
            if scores[i] < conf_thresh: continue
            
            label_idx = int(labels[i])
            if label_idx >= len(CLASSES): label_idx = 0
            
            color = COLORS[label_idx % len(COLORS)]
            label_text = CLASSES[label_idx]
            
            # --- ?ïÁ? Visible ??---
            if vis_masks is not None:
                img_vis = overlay_mask(img_vis, vis_masks[i], color, alpha=0.5, threshold=mask_thresh)
            
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            caption = f"{label_text} {scores[i]:.2f}"
            cv2.putText(img_vis, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --- ?ïÁ? Amodal ??---
            if amo_masks is not None:
                img_amo = overlay_mask(img_amo, amo_masks[i], color, alpha=0.3, threshold=mask_thresh)
                
            cv2.rectangle(img_amo, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_amo, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --- ?ïÁ? Invisible ??---
            if inv_masks is not None:
                img_inv = overlay_mask(img_inv, inv_masks[i], color, alpha=0.4, threshold=mask_thresh)
                
            cv2.rectangle(img_inv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_inv, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 6. Â≠òÊ?
        cv2.imwrite(os.path.join(vis_output_dir, img_name), img_vis)
        cv2.imwrite(os.path.join(amo_output_dir, img_name), img_amo)
        cv2.imwrite(os.path.join(inv_output_dir, img_name), img_inv)

    print(f"All done! Results saved to {output_folder}")

if __name__ == "__main__":
    # Ê®°Â?Ê¨äÈ?Ë∑ØÂ?
    ckpt = "output_pytorch/RegNet_3_2_DA_CP/model_final.pth"
    
    # [Ë®≠Â?] Ëº∏ÂÖ•Ë≥áÊ?Â§?(?ÖÂê´Ë¶ÅÊ∏¨Ë©¶Á??ñÁ?)
    input_dir = "./datasets/tomato/test_resized"
    
    # [Ë®≠Â?] Ëº∏Âá∫Ë≥áÊ?Â§?    output_dir = "demo_results"
    
    # [Ë®≠Â?] Backbone ?çÁ®±
    MY_BACKBONE = 'regnet_y_3_2gf'
    
    # ?∑Ë??πÊ¨°?®Ë?
    run_demo_folder(input_dir, output_dir, ckpt, backbone_name=MY_BACKBONE, conf_thresh=0.6, mask_thresh=0.3)

