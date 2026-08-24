# ==============================================================================
# Functional Block: Model Evaluation and Metrics Block
# Description: This module is responsible for the essential operations of Model Evaluation and Metrics Block.
# ==============================================================================
import torch
import os
import numpy as np
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
from model.model import AISFormerAmodal
from model.dataset import AmodalTomatoDataset, get_transform
import copy
import csv
from torchvision.ops import nms

# No inverse mapping needed for KINS

CAT_NAME_MAP = {}

def report_per_class_metrics(coco_eval, coco_gt, metric_type, csv_list, model_name=""):
    """
    è¨ˆç?ä¸¦å ±?Šæ??‹é??¥ç??‡æ?ï¼Œå??‚æ”¶?†åˆ° CSV ?—è¡¨ä¸?    """
    title = f"{metric_type} Metrics"
    print(f"\n{'-'*35} {title} {'-'*35}")
    
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = [c['name'] for c in cats]
    cat_ids = [c['id'] for c in cats]

    header = f"{'Category Name':<12} | {'AP':<6} | {'AP50':<6} | {'AP75':<6} | {'AR':<6} | {'Best P':<6} | {'Best R':<6} | {'F1':<6} | {'Best Score':<10}"
    print(header)
    print("-" * len(header))

    aps = []
    ap50s = []
    ap75s = []
    ars = []
    f1s = []
    best_scores = []

    for i, cat_id in enumerate(cat_ids):
        cat_name = CAT_NAME_MAP.get(cat_id, f"ID:{cat_id}")
        
        # ?–å?è©²é??¥ç?ç´¢å?ä½ç½® (??COCOeval ?§éƒ¨)
        try:
            cat_idx = i 
            
            # 1. ?–å? AP (IoU=0.5:0.95)
            # coco_eval.eval['precision'] shape: [T, R, K, A, M]
            # T: ious (0.5:0.05:0.95) -> 10 ??            # R: recalls (0:0.01:1) -> 101 ??            # K: categories
            # A: area ranges
            # M: max detections
            s = coco_eval.eval['precision'][:, :, cat_idx, 0, 2] # ??all area, maxDets=100
            if len(s[s > -1]) > 0:
                ap = np.mean(s[s > -1])
            else:
                ap = 0.0
                
            # 2. ?–å? AP50
            s_50 = coco_eval.eval['precision'][0, :, cat_idx, 0, 2]
            if len(s_50[s_50 > -1]) > 0:
                ap_50 = np.mean(s_50[s_50 > -1])
            else:
                ap_50 = 0.0

            # 2.5 ?–å? AP75
            s_75 = coco_eval.eval['precision'][5, :, cat_idx, 0, 2]
            if len(s_75[s_75 > -1]) > 0:
                ap_75 = np.mean(s_75[s_75 > -1])
            else:
                ap_75 = 0.0

            # 3. ?–å? AR
            s_ar = coco_eval.eval['recall'][:, cat_idx, 0, 2]
            if len(s_ar[s_ar > -1]) > 0:
                ar = np.mean(s_ar[s_ar > -1])
            else:
                ar = 0.0

            # 4. å°‹æ‰¾?€ä½?F1 (?ºæ–¼ PR ?²ç?)
            # ?‘å€‘ç? IoU=0.5 ??PR ?²ç?
            p_curve = coco_eval.eval['precision'][0, :, cat_idx, 0, 2] # shape (101,)
            r_axis = np.linspace(0, 1, 101)
            
            f1_curve = 2 * p_curve * r_axis / (p_curve + r_axis + 1e-10)
            best_f1_idx = np.argmax(f1_curve)
            best_f1 = f1_curve[best_f1_idx]
            best_p = p_curve[best_f1_idx]
            best_r = r_axis[best_f1_idx]
            
            # 5. ?€ä½³å???(å¾?COCOeval ä¸­æ???
            has_scores = 'scores' in coco_eval.eval
            if has_scores:
                best_score = coco_eval.eval['scores'][0, best_f1_idx, cat_idx, 0, 2]
            else:
                best_score = 0.0

            print(f"{cat_name:<12} | {ap:.4f} | {ap_50:.4f} | {ap_75:.4f} | {ar:.4f} | {best_p:.4f} | {best_r:.4f} | {best_f1:.4f} | {best_score:.4f}")
            
            row_data = {
                "Model": f"{model_name}_v1_maxsize3000",
                "Metric Type": metric_type,
                "Category Name": cat_name,
                "AP": f"{ap:.4f}",
                "AP50": f"{ap_50:.4f}",
                "AP75": f"{ap_75:.4f}",
                "AR": f"{ar:.4f}",
                "Best P": f"{best_p:.4f}",
                "Best R": f"{best_r:.4f}",
                "F1": f"{best_f1:.4f}",
                "Best Score": f"{best_score:.4f}"
            }
            
            aps.append(ap)
            ap50s.append(ap_50)
            ap75s.append(ap_75)
            ars.append(ar)
            f1s.append(best_f1)
            best_scores.append(best_score)
        except Exception as e:
            print(f"{cat_name:<12} | Error: {e}")
            row_data = {
            "Model": f"{model_name}_v1_maxsize3000",
            "Metric Type": metric_type, "Category Name": cat_name, "AP": "0.0"}
            
        csv_list.append(row_data)

    print("-" * len(header))
    if aps:
        mean_ap = np.mean(aps) if aps else 0
        mean_ap50 = np.mean(ap50s) if ap50s else 0
        mean_ap75 = np.mean(ap75s) if ap75s else 0
        mean_ar = np.mean(ars) if ars else 0
        mean_f1 = np.mean(f1s) if f1s else 0
        mean_score = np.mean(best_scores) if best_scores else 0
        
        print(f"{'MEAN':<12} | {mean_ap:.4f} | {mean_ap50:.4f} | {mean_ap75:.4f} | {mean_ar:.4f} | {'-':<6} | {'-':<6} | {mean_f1:.4f} | {mean_score:.4f}")
        
        csv_list.append({
            "Model": f"{model_name}_v1_maxsize3000",
            "Metric Type": metric_type,
            "Category Name": "MEAN",
            "AP": f"{mean_ap:.4f}",
            "AP50": f"{mean_ap50:.4f}",
            "AP75": f"{mean_ap75:.4f}",
            "AR": f"{mean_ar:.4f}",
            "Best P": "-",
            "Best R": "-",
            "F1": f"{mean_f1:.4f}",
            "Best Score": f"{mean_score:.4f}"
        })
        
    print("=" * len(header) + "\n")

def run_evaluation(args):
    device = torch.device(args.device)
    csv_results = []
    
    print(f"Loading model from {args.checkpoint}...")
    print(f"Using Backbone: {args.backbone}")
    
    model = AISFormerAmodal(
        num_classes=9,
        min_size=800,
        max_size=3000,
        backbone_name=args.backbone,
        bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
        use_light_mask_head=True,
        official_loss_weight=False
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 2. æº–å? Dataset
    dataset = AmodalTomatoDataset(args.ann_file, args.img_dir, get_transform(train=False))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x))
    )

    print("Running inference on all images...")
    results_vis = []
    results_amo = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            try:
                images = list(img.to(device) for img in images)
                outputs = model(images)
                outputs = [{k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in out.items()} for out in outputs]
                torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                print(f"? ï? OOM on image {targets[0]['image_id'].item()}, skipping.")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
            except Exception as e:
                print(f"? ï? Unexpected error on image {targets[0]['image_id'].item()}: {e}, skipping.")
                continue

            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                
                scores = output["scores"]
                labels = output["labels"]
                boxes = output["boxes"]

                if len(scores) > 0:
                    agnostic_keep_idxs = nms(boxes, scores, iou_threshold=0.95)
                    scores = scores[agnostic_keep_idxs]
                    labels = labels[agnostic_keep_idxs]
                    boxes = boxes[agnostic_keep_idxs]

                    if "pred_visible_masks" in output:
                        output["pred_visible_masks"] = output["pred_visible_masks"][agnostic_keep_idxs]
                    if "pred_amodal_masks" in output:
                        output["pred_amodal_masks"] = output["pred_amodal_masks"][agnostic_keep_idxs]
                
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                boxes = boxes.cpu().numpy()
                
                keep_idxs = scores > 0.001 
                scores = scores[keep_idxs]
                labels = labels[keep_idxs]
                boxes = boxes[keep_idxs]
                
                if len(scores) == 0:
                    continue

                if "pred_visible_masks" in output:
                    vis_masks = output["pred_visible_masks"][keep_idxs].squeeze(1).cpu().numpy() > 0.5
                    vis_masks_transposed = np.asfortranarray(vis_masks.transpose(1, 2, 0).astype(np.uint8))
                    vis_rles = mask_util.encode(vis_masks_transposed)
                    for rle in vis_rles: rle['counts'] = rle['counts'].decode('utf-8')
                else:
                    vis_rles = [None] * len(scores)

                if "pred_amodal_masks" in output:
                    amo_masks = output["pred_amodal_masks"][keep_idxs].squeeze(1).cpu().numpy() > 0.5
                    amo_masks_transposed = np.asfortranarray(amo_masks.transpose(1, 2, 0).astype(np.uint8))
                    amo_rles = mask_util.encode(amo_masks_transposed)
                    for rle in amo_rles: rle['counts'] = rle['counts'].decode('utf-8')
                else:
                    amo_rles = [None] * len(scores)

                for i in range(len(scores)):
                    pred_label = int(labels[i])
                    category_id = int(pred_label)
                    box = boxes[i]
                    amodal_bbox_wh = [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])]
                    
                    if amo_rles[i] is not None:
                        results_amo.append({
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": amodal_bbox_wh,
                            "score": float(scores[i]),
                            "segmentation": amo_rles[i]
                        })

                    if vis_rles[i] is not None:
                        vis_bbox = mask_util.toBbox(vis_rles[i])
                        results_vis.append({
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [float(x) for x in vis_bbox],
                            "score": float(scores[i]),
                            "segmentation": vis_rles[i]
                        })

    coco_gt = COCO(args.ann_file)
    
    if results_vis:
        print(f"\n{'='*20} [1/4] Evaluating Visible Masks {'='*20}")
        coco_gt_vis = copy.deepcopy(coco_gt)
        for ann_id, ann in coco_gt_vis.anns.items():
            if 'inmodal_seg' in ann:
                ann['segmentation'] = ann['inmodal_seg']

        coco_dt_vis = coco_gt_vis.loadRes(results_vis)
        coco_eval_vis = COCOeval(coco_gt_vis, coco_dt_vis, 'segm')
        coco_eval_vis.evaluate(); coco_eval_vis.accumulate(); coco_eval_vis.summarize()
        report_per_class_metrics(coco_eval_vis, coco_gt_vis, "Visible Mask", csv_results, args.backbone)

    if results_vis:
        print(f"\n{'='*20} [2/4] Evaluating Visible BBoxes {'='*20}")
        coco_gt_vis_box = copy.deepcopy(coco_gt)
        for ann_id, ann in coco_gt_vis_box.anns.items():
            if 'visible_bbox' in ann: ann['bbox'] = ann['visible_bbox']
            elif 'inmodal_bbox' in ann: ann['bbox'] = ann['inmodal_bbox']
        
        coco_dt_vis = coco_gt_vis_box.loadRes(results_vis)
        coco_eval_bbox = COCOeval(coco_gt_vis_box, coco_dt_vis, 'bbox') 
        coco_eval_bbox.evaluate(); coco_eval_bbox.accumulate(); coco_eval_bbox.summarize()
        report_per_class_metrics(coco_eval_bbox, coco_gt_vis_box, "Visible BBox", csv_results, args.backbone)

    if results_amo:
        print(f"\n{'='*20} [3/4] Evaluating Amodal Masks {'='*20}")
        coco_gt_amo = copy.deepcopy(coco_gt)
        for ann_id, ann in coco_gt_amo.anns.items():
            if 'i_segmentation' in ann:
                ann['segmentation'] = ann['i_segmentation']

        coco_dt_amo = coco_gt_amo.loadRes(results_amo)
        coco_eval_amo = COCOeval(coco_gt_amo, coco_dt_amo, 'segm')
        coco_eval_amo.evaluate(); coco_eval_amo.accumulate(); coco_eval_amo.summarize()
        report_per_class_metrics(coco_eval_amo, coco_gt_amo, "Amodal Mask", csv_results, args.backbone)

    if results_amo:
        print(f"\n{'='*20} [4/4] Evaluating Amodal BBoxes {'='*20}")
        coco_gt_box = copy.deepcopy(coco_gt)
        for ann_id, ann in coco_gt_box.anns.items():
            if 'amodal_bbox' in ann: ann['bbox'] = ann['amodal_bbox']
        
        coco_dt_box = coco_gt_box.loadRes(results_amo)
        coco_eval_box = COCOeval(coco_gt_box, coco_dt_box, 'bbox') 
        coco_eval_box.evaluate(); coco_eval_box.accumulate(); coco_eval_box.summarize()
        report_per_class_metrics(coco_eval_box, coco_gt_box, "Amodal BBox", csv_results, args.backbone)

    if csv_results and args.csv_path:
        print(f"\n?? Writing results to {args.csv_path} ...")
        fields = ["Model", "Metric Type", "Category Name", "AP", "AP50", "AP75", "AR", "Best P", "Best R", "F1", "Best Score"]
        file_exists = os.path.isfile(args.csv_path)
        with open(args.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists: writer.writeheader()
            writer.writerows(csv_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output/kins_2026/run_official/resnet50_DA_Run/model_best.pth")
    parser.add_argument("--ann_file", default="datasets/KINS/instances_val.json")
    parser.add_argument("--img_dir", default="datasets/KINS/testing/image_2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--csv_path", default="output/kins_2026/run_official/resnet50_DA_Run/kins_val_results.csv")
    parser.add_argument("--use_light_mask_head", action="store_true", help="Use the official lightweight mask head")
    run_evaluation(parser.parse_args())

