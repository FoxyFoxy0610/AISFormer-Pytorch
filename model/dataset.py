# ==============================================================================
# Functional Block: Dataset Loading and Preprocessing Block
# Description: This module is responsible for the essential operations of Dataset Loading and Preprocessing Block.
# ==============================================================================
import torch
import os
import numpy as np
from PIL import Image, ImageOps
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import random 
from torchvision.transforms import v2 as T

# ?ºç?è½‰æ?
def get_transform(train):
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True)) 
    return T.Compose(transforms)

class AmodalTomatoDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None, target_size=3000):
        self.img_dir = img_dir
        self.transforms = transforms
        self.coco = COCO(annotations_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.target_size = target_size
        self.is_train = "train" in annotations_file.lower()
        print(f"?? Dataset initialized. Mode: CPU (Pure PIL) Resize to {target_size}, is_train={self.is_train}")

    def _polygons_to_mask(self, segm, height, width):
        """Helper: å°?Polygon/RLE è½‰ç‚º Numpy array (uint8)"""
        # [?œéµä¿®æ­£] ?™è£¡?Ÿæœ¬?žå‚³ torch.zerosï¼Œå???PIL å´©æ½°
        # ?¾åœ¨çµ±ä??žå‚³ np.zeros
        
        if isinstance(segm, list):
            if not segm: 
                return np.zeros((height, width), dtype=np.uint8)
            rles = mask_util.frPyObjects(segm, height, width)
            rle = mask_util.merge(rles)
        elif isinstance(segm, dict):
            rle = segm
        else:
            return np.zeros((height, width), dtype=np.uint8)
        
        mask = mask_util.decode(rle)
        # mask ?¬èº«å°±æ˜¯ numpy array
        return mask

    def _resize_sample_cpu(self, data):
        """ä½¿ç”¨ç´?PIL ?²è??€?‰ç¸®??(?€ç©©å?ï¼Œé¿??Illegal Instruction)"""
        w, h = data['width'], data['height']
        scale = self.target_size / max(w, h)
        
        if scale >= 1.0:
            # ä¸é?ç¸®æ”¾ï¼Œç›´?¥è? Tensor
            img_tensor = T.functional.to_image(data['image'])
            img_tensor = T.functional.to_dtype(img_tensor, torch.float32, scale=True)
            data['image'] = img_tensor
            
            # å°?masks (numpy list) è½‰ç‚º Tensor stack
            # ?™è£¡å¿…é???np.stackï¼Œå???list è£¡é¢?¾åœ¨?¨æ˜¯ numpy array
            data['vis_masks'] = torch.from_numpy(np.stack(data['vis_masks'])) if data['vis_masks'] else torch.zeros((0, h, w), dtype=torch.uint8)
            data['amodal_masks'] = torch.from_numpy(np.stack(data['amodal_masks'])) if data['amodal_masks'] else torch.zeros((0, h, w), dtype=torch.uint8)
            data['bg_masks'] = torch.from_numpy(np.stack(data['bg_masks'])) if data['bg_masks'] else torch.zeros((0, h, w), dtype=torch.uint8)
            return data
            
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 1. Image Resize (PIL Bilinear)
        img_pil = data['image'].resize((new_w, new_h), resample=Image.BILINEAR)
        img_tensor = T.functional.to_image(img_pil)
        img_tensor = T.functional.to_dtype(img_tensor, torch.float32, scale=True)
        data['image'] = img_tensor 
        
        # 2. Mask Resize (PIL Nearest)
        def resize_stack_pil(masks_list):
            if not masks_list: 
                return torch.zeros((0, new_h, new_w), dtype=torch.uint8)
            
            resized_masks = []
            for mask_np in masks_list:
                # Numpy -> PIL (?™è£¡?¾åœ¨å®‰å…¨äº†ï?? ç‚º mask_np ç¢ºä¿¡??numpy)
                mask_pil = Image.fromarray(mask_np)
                # Resize (Nearest to keep 0/1)
                mask_pil = mask_pil.resize((new_w, new_h), resample=Image.NEAREST)
                # PIL -> Tensor
                resized_masks.append(torch.from_numpy(np.array(mask_pil)))
            
            return torch.stack(resized_masks)

        data['vis_masks'] = resize_stack_pil(data['vis_masks'])
        data['amodal_masks'] = resize_stack_pil(data['amodal_masks'])
        data['bg_masks'] = resize_stack_pil(data['bg_masks'])
        
        # 3. BBox Resize
        if data['boxes'].numel() > 0: 
            data['boxes'] *= scale
            
        data['width'] = new_w
        data['height'] = new_h
        return data

    def _load_sample(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        
        img_raw = Image.open(path).convert("RGB")
        try:
            img = ImageOps.exif_transpose(img_raw)
        except:
            img = img_raw

        img_w, img_h = img.size 
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        valid_anns = []
        for ann in anns:
            if 'segmentation' not in ann: continue
            if ann.get('iscrowd', 0) == 1: continue
            valid_anns.append(ann)
            
        boxes, labels, vis_masks, amodal_masks, bg_masks = [], [], [], [], []
        if valid_anns:
            for ann in valid_anns:
                boxes.append(ann['bbox'])
                labels.append(ann['category_id'])
                vis_masks.append(self._polygons_to_mask(ann.get('inmodal_seg', ann['segmentation']), img_h, img_w))
                amodal_masks.append(self._polygons_to_mask(ann['segmentation'], img_h, img_w))

            # Dynamically compute bg_masks (occluder masks) for each object
            # An occluder mask for object `i` is the union of the visible masks of all OTHER objects.
            # Using O(N) integer sum optimization:
            total_vis = np.sum(vis_masks, axis=0) if len(vis_masks) > 0 else np.zeros((img_h, img_w))
            for v in vis_masks:
                bg = (total_vis - v) > 0
                bg_masks.append(bg.astype(np.uint8))

            boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes_t[:, 2:] += boxes_t[:, :2] 
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            # masks ?¾åœ¨??list of numpy arrays
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)
            vis_masks = []
            amodal_masks = []
            bg_masks = []

        return {
            "image": img, 
            "boxes": boxes_t, "labels": labels_t,
            "vis_masks": vis_masks, "amodal_masks": amodal_masks, "bg_masks": bg_masks,
            "height": img_h, "width": img_w
        }

    def __getitem__(self, idx):
        data = self._load_sample(idx)
        
        # Horizontal Flip (DA)
        if self.is_train and random.random() > 0.5:
            # 1. Flip Image
            data["image"] = data["image"].transpose(Image.FLIP_LEFT_RIGHT)
            # 2. Flip Boxes
            if data["boxes"].numel() > 0:
                w = data["width"]
                xmin = data["boxes"][:, 0].clone()
                xmax = data["boxes"][:, 2].clone()
                data["boxes"][:, 0] = w - xmax
                data["boxes"][:, 2] = w - xmin
            # 3. Flip Masks
            for i in range(len(data["vis_masks"])):
                data["vis_masks"][i] = np.fliplr(data["vis_masks"][i])
            for i in range(len(data["amodal_masks"])):
                data["amodal_masks"][i] = np.fliplr(data["amodal_masks"][i])
            for i in range(len(data["bg_masks"])):
                data["bg_masks"][i] = np.fliplr(data["bg_masks"][i])

        data = self._resize_sample_cpu(data) 
        
        target = {}
        target["image_id"] = torch.tensor([self.ids[idx]])
        target["height"] = torch.tensor(data['height'])
        target["width"] = torch.tensor(data['width'])
        target["boxes"] = data["boxes"]
        target["labels"] = data["labels"]
        target["gt_visible_masks"] = data["vis_masks"]
        target["gt_amodal_masks"] = data["amodal_masks"]
        target["gt_background_objs_masks"] = data["bg_masks"]

        if target["boxes"].numel() > 0:
            target["boxes"][:, 0].clamp_(0, data['width'])
            target["boxes"][:, 1].clamp_(0, data['height'])
            target["boxes"][:, 2].clamp_(0, data['width'])
            target["boxes"][:, 3].clamp_(0, data['height'])

        return data['image'], target

    def __len__(self):
        return len(self.ids)
