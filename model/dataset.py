# ==============================================================================
# Functional Block: Dataset Loading and Preprocessing Block
# Description: This module is responsible for the essential operations of Dataset Loading and Preprocessing Block.
# ==============================================================================
import torch
import os
import numpy as np
from PIL import Image, ImageOps
import cv2  # [?啣?] OpenCV 擃葬??from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import random 
from torchvision.transforms import v2 as T
import torch.nn.functional as F  # [?啣?] PyTorch C++ ?桃蔗蝮格

# ?箇?頧?
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
        print(f"?? KINS Dataset initialized. Mode: C++/OpenCV Accelerated. Target Resize: {target_size}, is_train={self.is_train}")

    def _poly_to_rle(self, segm, height, width):
        """[?芸?] ?? Polygon 頧 RLE dict嚗??脰? decode嚗誑靘踹?蝥?C++ ?寥?閫?Ⅳ"""
        if isinstance(segm, list):
            if not segm: 
                return mask_util.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
            rles = mask_util.frPyObjects(segm, height, width)
            return mask_util.merge(rles)
        elif isinstance(segm, dict):
            return segm
        else:
            return mask_util.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))

    def _resize_sample_cpu(self, data):
        """[?芸?] 雿輻 OpenCV ??PyTorch C++ Backend ?脰?擃葬??""
        w, h = data['width'], data['height']
        scale = self.target_size / max(w, h)
        
        # 頧???Numpy array 敺耦???(H, W, 3)嚗???靘??RGB
        img_np = np.array(data['image'])
        
        if scale >= 1.0:
            img_tensor = T.functional.to_image(img_np)
            img_tensor = T.functional.to_dtype(img_tensor, torch.float32, scale=True)
            data['image'] = img_tensor
            
            data['vis_masks'] = torch.from_numpy(data['vis_masks']) if len(data['vis_masks']) > 0 else torch.zeros((0, h, w), dtype=torch.uint8)
            data['amodal_masks'] = torch.from_numpy(data['amodal_masks']) if len(data['amodal_masks']) > 0 else torch.zeros((0, h, w), dtype=torch.uint8)
            data['bg_masks'] = torch.from_numpy(data['bg_masks']) if len(data['bg_masks']) > 0 else torch.zeros((0, h, w), dtype=torch.uint8)
            return data
            
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 1. Image Resize (OpenCV BICUBIC)
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        img_tensor = T.functional.to_image(img_resized)
        img_tensor = T.functional.to_dtype(img_tensor, torch.float32, scale=True)
        data['image'] = img_tensor 
        
        # 2. Mask Resize (PyTorch C++ Backend: F.interpolate)
        def resize_masks_tensor(masks_np):
            if len(masks_np) == 0:
                return torch.zeros((0, new_h, new_w), dtype=torch.uint8)
            # 撠?Numpy shape (N, H, W) 頧 Tensor (N, 1, H, W)
            masks_t = torch.from_numpy(masks_np).unsqueeze(1).float()
            # C++ ?寥??葬??            masks_resized = F.interpolate(masks_t, size=(new_h, new_w), mode='nearest')
            return masks_resized.squeeze(1).byte()

        data['vis_masks'] = resize_masks_tensor(data['vis_masks'])
        data['amodal_masks'] = resize_masks_tensor(data['amodal_masks'])
        data['bg_masks'] = resize_masks_tensor(data['bg_masks'])
        
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
            
        boxes, labels, vis_rles, amodal_rles = [], [], [], []
        if valid_anns:
            for ann in valid_anns:
                boxes.append(ann['bbox'])
                labels.append(ann['category_id'])
                # KINS 璅酉?摩嚗nmodal_seg ?臬閬蝵抬?segmentation ??Amodal ?桃蔗
                vis_rles.append(self._poly_to_rle(ann.get('inmodal_seg', ann['segmentation']), img_h, img_w))
                amodal_rles.append(self._poly_to_rle(ann['segmentation'], img_h, img_w))

            # [?芸?] Pycocotools ?寥? C++ 閫?Ⅳ
            vis_masks_hw_n = mask_util.decode(vis_rles)
            amodal_masks_hw_n = mask_util.decode(amodal_rles)

            # [?芸?] ?? Occluder 閮? (?湔??(H, W, N) 銝誨?剝?蝞?
            total_vis = vis_masks_hw_n.sum(axis=2, keepdims=True)
            bg_masks_hw_n = (total_vis - vis_masks_hw_n) > 0

            # 頧蔭??(N, H, W)
            vis_masks = vis_masks_hw_n.transpose(2, 0, 1)
            amodal_masks = amodal_masks_hw_n.transpose(2, 0, 1)
            bg_masks = bg_masks_hw_n.transpose(2, 0, 1).astype(np.uint8)

            boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes_t[:, 2:] += boxes_t[:, :2] 
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)
            vis_masks = np.zeros((0, img_h, img_w), dtype=np.uint8)
            amodal_masks = np.zeros((0, img_h, img_w), dtype=np.uint8)
            bg_masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

        return {
            "image": img, 
            "boxes": boxes_t, "labels": labels_t,
            "vis_masks": vis_masks, "amodal_masks": amodal_masks, "bg_masks": bg_masks,
            "height": img_h, "width": img_w
        }

    def __getitem__(self, idx):
        data = self._load_sample(idx)
        
        # [?芸?] Horizontal Flip (DA) - ?寧???蕃頧?        if self.is_train and random.random() > 0.5:
            # 1. Flip Image
            data["image"] = data["image"].transpose(Image.FLIP_LEFT_RIGHT)
            # 2. Flip Boxes
            if data["boxes"].numel() > 0:
                w = data["width"]
                xmin = data["boxes"][:, 0].clone()
                xmax = data["boxes"][:, 2].clone()
                data["boxes"][:, 0] = w - xmax
                data["boxes"][:, 2] = w - xmin
            # 3. Flip Masks (雿輻 np.flip ?寥??? (N, H, W) ?祝摨衣雁摨佗???axis=2)
            if len(data["vis_masks"]) > 0:
                data["vis_masks"] = np.flip(data["vis_masks"], axis=2).copy()
                data["amodal_masks"] = np.flip(data["amodal_masks"], axis=2).copy()
                data["bg_masks"] = np.flip(data["bg_masks"], axis=2).copy()

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
