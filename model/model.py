# ==============================================================================
# Functional Block: Main Model Architecture Block
# Description: This module is responsible for the essential operations of Main Model Architecture Block.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
from typing import Optional, List, Dict, Tuple
from collections import OrderedDict
import numpy as np
import pycocotools.mask as mask_util 

from torchvision.ops import MultiScaleRoIAlign, nms, roi_align
import torchvision.ops.poolers as poolers

class AlignedMultiScaleRoIAlign(MultiScaleRoIAlign):
    """
    A wrapper around MultiScaleRoIAlign that enforces aligned=True for PyTorch/Torchvision,
    which matches Detectron2's ROIAlignV2 precision by shifting pixels by -0.5.
    """
    def forward(self, x, boxes, image_shapes):
        original_roi_align = poolers.roi_align
        def aligned_roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=True):
            return original_roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned=True)
        poolers.roi_align = aligned_roi_align
        try:
            return super().forward(x, boxes, image_shapes)
        finally:
            poolers.roi_align = original_roi_align
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads, paste_masks_in_image

# ==============================================================================
# Model Loading Utilities (Safe Imports for Backbone)
# ==============================================================================
from torchvision.models import get_model, ResNet50_Weights, ResNet101_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# FPN Compatibility
from torchvision.ops import FeaturePyramidNetwork 
try:
    from torchvision.ops import LastLevelMaxPool
except ImportError:
    try:
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
    except ImportError:
        try:
            from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
        except ImportError:
            class ExtraFPNBlock(torch.nn.Module): pass

        class LastLevelMaxPool(ExtraFPNBlock):
            def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
                names.append("pool")
                y.append(F.max_pool2d(y[-1], 1, 2, 0))
                return y, names

from .transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from .position_encoding import PositionEmbeddingLearned
from .mlp import MLP
import timm

# ==============================================================================
# SECTION 1: PREDICTOR
# ==============================================================================

class FastRCNNCustomPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.dim() == 4: x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

# ==============================================================================
# SECTION 2: HEAD
# ==============================================================================

class AISFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes, pooler_resolution, dim_feedforward=2048, n_heads=2, n_layers=1, use_light_mask_head=False, official_loss_weight=False, **kwargs):
        super().__init__()
        self.official_loss_weight = official_loss_weight
        self.pooler = AlignedMultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=pooler_resolution, sampling_ratio=2)
        
        self.conv_dim = 256
        if use_light_mask_head:
            self.conv_backbone = nn.Sequential(
                nn.Conv2d(in_channels, self.conv_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.conv_dim, self.conv_dim, 2, stride=2), 
                nn.ReLU(inplace=True),
                nn.Conv2d(self.conv_dim, self.conv_dim, 1)
            )
        else:
            self.conv_backbone = nn.Sequential(
                nn.Conv2d(in_channels, self.conv_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.conv_dim, self.conv_dim, 2, stride=2), 
                nn.ReLU(inplace=True),
                nn.Conv2d(self.conv_dim, self.conv_dim, 1)
            )

        emb_dim = self.conv_dim
        self.positional_encoding = PositionEmbeddingLearned(emb_dim // 2)
        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.n_output_masks = 4 
        self.query_embed = nn.Embedding(num_embeddings=self.n_output_masks, embedding_dim=emb_dim)
        self.mask_embed_mlp = MLP(emb_dim, emb_dim, emb_dim, 3) 
        self.subtract_model = MLP(emb_dim * 2, emb_dim, emb_dim, 2) 
        self.norm_rois = nn.LayerNorm(emb_dim)
        if use_light_mask_head:
            self.pixel_embed_layer = nn.Conv2d(self.conv_dim, self.conv_dim, 1)
        else:
            self.pixel_embed_layer = nn.Sequential(
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.conv_dim, self.conv_dim, 1)
            )

    def forward(self, features, proposals, image_shapes):
        x = self.pooler(features, proposals, image_shapes) 
        x_short = self.conv_backbone(x) 
        bs = x_short.shape[0]
        
        feat_embs = x_short.flatten(2).permute(2, 0, 1) 
        pos_embed = self.positional_encoding.forward_tensor(x_short)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        encoded_feat_embs = self.transformer_encoder(feat_embs, pos=pos_embed)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) 
        tgt = torch.zeros_like(query_embed)
        
        decoder_output = self.transformer_decoder(tgt, encoded_feat_embs, pos=pos_embed, query_pos=query_embed) 
        decoder_output = decoder_output.squeeze(0).transpose(0, 1) 

        mask_embs = self.mask_embed_mlp(decoder_output) 
        combined_feat = torch.cat([mask_embs[:, 2, :], mask_embs[:, 0, :]], dim=1) 
        invisible_embs = self.subtract_model(combined_feat).unsqueeze(1) 
        final_queries = torch.cat([mask_embs[:, :3, :], invisible_embs], dim=1) 

        roi_embedding = encoded_feat_embs.permute(1, 2, 0).unflatten(-1, (28, 28))
        roi_embedding = roi_embedding + x_short 
        roi_embedding = self.norm_rois(roi_embedding.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        pixel_embeddings = self.pixel_embed_layer(roi_embedding)

        outputs_mask = torch.einsum("bqc,bchw->bqhw", final_queries, pixel_embeddings)

        return { 
            "visible": outputs_mask[:, 0:1],       
            "background_obj": outputs_mask[:, 1:2], 
            "amodal": outputs_mask[:, 2:3],        
            "invisible": outputs_mask[:, 3:4]      
        }

# ==============================================================================
# SECTION 3: ROI HEADS (Corrected Loss Calculation)
# ==============================================================================
class TimmBackboneWithFPN(nn.Module):
    """Â∞àÈ??ïÁ? timm Ê®°Â???FPN ?ÖË???""
    def __init__(self, backbone, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        features = self.body(x)
        # Â∞?timm ?ûÂÇ≥??List ËΩâÁÇ∫ OrderedDict ‰æ?FPN ‰ΩøÁî®
        out = OrderedDict([(str(i), f) for i, f in enumerate(features)])
        out = self.fpn(out)
        return out

class AISFormerRoIHeads(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, mask_head, score_thresh, nms_thresh, detections_per_img, **kwargs):
        kwargs.pop("mask_rasterization_chunk_size", None)
        super().__init__(box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh, nms_thresh, detections_per_img, **kwargs)
        self.mask_head = mask_head

    @staticmethod
    def segm_to_mask(segm, height: int, width: int, device) -> torch.Tensor:
        if isinstance(segm, list):
            rles = mask_util.frPyObjects(segm, height, width)
            rle = mask_util.merge(rles)
        elif isinstance(segm, dict):
            rle = segm
        else:
            raise TypeError(f"Unsupported segmentation type: {type(segm)}")
        mask = mask_util.decode(rle)
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=device)
        return mask.unsqueeze(0)

    def _get_aligned_preds_and_gts(self, targets, labels, matched_idxs, proposals, all_mask_preds_for_type, pos_labels_cat, mask_type_gt_key, device, num_classes, image_shapes, is_invisible=False):
        aligned_preds_list = []
        cropped_gt_list = []
        mask_side_len = all_mask_preds_for_type.size(2)
        proposal_start_idx = 0
        key_map = {'gt_visible_segms': 'gt_visible_masks', 'gt_amodal_segms': 'gt_amodal_masks', 'gt_background_objs_segms': 'gt_background_objs_masks'}
        real_key = key_map.get(mask_type_gt_key, mask_type_gt_key)

        for i in range(len(targets)): 
            pos_inds_in_image = torch.where(labels[i] > 0)[0]
            num_pos_in_image = pos_inds_in_image.numel()
            if num_pos_in_image == 0: continue
            
            if is_invisible:
                has_vis = targets[i].get('gt_visible_masks') is not None and len(targets[i]['gt_visible_masks']) > 0
                has_amodal = targets[i].get('gt_amodal_masks') is not None and len(targets[i]['gt_amodal_masks']) > 0
                if not (has_vis and has_amodal):
                    proposal_start_idx += num_pos_in_image
                    continue
            else:
                if targets[i].get(real_key) is None or len(targets[i][real_key]) == 0:
                    proposal_start_idx += num_pos_in_image
                    continue

            proposals_for_image = proposals[i][pos_inds_in_image]
            gt_inds_for_pos_proposals = matched_idxs[i][pos_inds_in_image]
            
            orig_h = targets[i]["height"].item()
            orig_w = targets[i]["width"].item()
            resized_h, resized_w = image_shapes[i]
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h
            
            proposals_scaled = proposals_for_image.clone()
            proposals_scaled[:, 0::2] *= scale_x
            proposals_scaled[:, 1::2] *= scale_y

            unique_gt_ids = torch.unique(gt_inds_for_pos_proposals)
            cropped_gt_for_image = torch.zeros((num_pos_in_image, mask_side_len, mask_side_len), dtype=torch.float32, device=device)
            
            for gt_id in unique_gt_ids:
                mask_inds = torch.where(gt_inds_for_pos_proposals == gt_id)[0]
                proposals_group = proposals_scaled[mask_inds] 
                
                if is_invisible:
                    vis_mask = targets[i]['gt_visible_masks'][gt_id]
                    amodal_mask = targets[i]['gt_amodal_masks'][gt_id]
                    current_gt_mask = (vis_mask ^ amodal_mask).to(device=device, dtype=torch.float32)
                else:
                    current_gt_mask = targets[i][real_key][gt_id].to(device=device, dtype=torch.float32)
                
                cropped_group = roi_align(
                    current_gt_mask.view(1, 1, int(orig_h), int(orig_w)), 
                    [proposals_group],
                    (mask_side_len, mask_side_len),
                    1.0, -1,
                    aligned=True
                )
                cropped_gt_for_image[mask_inds] = cropped_group.squeeze(1)

            cropped_gt_list.append(cropped_gt_for_image)
            
            batch_indices = (proposal_start_idx + torch.arange(num_pos_in_image, device=device))
            preds_for_image = all_mask_preds_for_type[batch_indices] 
            aligned_preds_list.append(preds_for_image.squeeze(1))
            proposal_start_idx += num_pos_in_image

        if not cropped_gt_list: return None, None
        return torch.cat(aligned_preds_list, dim=0), torch.cat(cropped_gt_list, dim=0)

    def forward(self, features, proposals, image_shapes, targets=None):
        # 1. ÁØ©ÈÅ∏ Proposal
        if self.training:
            valid_proposals = []
            for p in proposals:
                keep = (p[:, 2] > p[:, 0] + 1e-4) & (p[:, 3] > p[:, 1] + 1e-4)
                if keep.sum() > 0: valid_proposals.append(p[keep])
                else: valid_proposals.append(p)
            proposals = valid_proposals
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels, regression_targets, matched_idxs = None, None, None

        # 2. Box Head Forward
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        
        # 3. Training Logic (Loss Calculation)
        if self.training:
            num_classes = class_logits.shape[-1]
            loss_classifier = F.cross_entropy(class_logits, torch.cat(labels, dim=0))
            
            sampled_pos_inds_subset = torch.where(torch.cat(labels, dim=0) > 0)[0]
            labels_pos = torch.cat(labels, dim=0)[sampled_pos_inds_subset]
            
            box_regression = box_regression.reshape(box_regression.size(0), num_classes, 4)
            box_regression_pos = box_regression[sampled_pos_inds_subset]
            regression_targets_pos = torch.cat(regression_targets, dim=0)[sampled_pos_inds_subset]
            
            if labels_pos.numel() > 0:
                box_pred = box_regression_pos[torch.arange(labels_pos.size(0), device=labels_pos.device), labels_pos]
                valid_mask = torch.isfinite(regression_targets_pos).all(dim=1)
                if not valid_mask.all():
                    box_pred = box_pred[valid_mask]
                    regression_targets_pos = regression_targets_pos[valid_mask]
                
                loss_box_reg = F.smooth_l1_loss(box_pred, regression_targets_pos, beta=1.0, reduction="mean") if box_pred.numel() > 0 else torch.tensor(0.0, device=class_logits.device)
            else:
                loss_box_reg = torch.tensor(0.0, device=class_logits.device)
            
            losses.update({"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg})

            # [?úÈçµ‰øÆÂæ©] Mask Loss Calculation
            pos_proposals = [p[torch.where(l > 0)[0]] for p, l in zip(proposals, labels)]
            pos_proposals_cat = torch.cat(pos_proposals, dim=0)

            if pos_proposals_cat.numel() > 0:
                mask_predictions = self.mask_head(features, pos_proposals, image_shapes)
                pos_labels_cat = torch.cat(labels, dim=0)[torch.where(torch.cat(labels, dim=0) > 0)[0]]
                device = class_logits.device
                
                # Ë®àÁ??õÁ®Æ Mask ??Loss
                pred_a, gt_a = self._get_aligned_preds_and_gts(targets, labels, matched_idxs, proposals, mask_predictions["amodal"], pos_labels_cat, 'gt_amodal_segms', device, num_classes, image_shapes)
                pred_v, gt_v = self._get_aligned_preds_and_gts(targets, labels, matched_idxs, proposals, mask_predictions["visible"], pos_labels_cat, 'gt_visible_segms', device, num_classes, image_shapes)
                pred_b, gt_b = self._get_aligned_preds_and_gts(targets, labels, matched_idxs, proposals, mask_predictions["background_obj"], pos_labels_cat, 'gt_background_objs_segms', device, num_classes, image_shapes)
                pred_i, gt_i = self._get_aligned_preds_and_gts(targets, labels, matched_idxs, proposals, mask_predictions["invisible"], pos_labels_cat, '', device, num_classes, image_shapes, is_invisible=True)

                for name, (pred, gt) in [("loss_a_mask", (pred_a, gt_a)), ("loss_vi_mask", (pred_v, gt_v)), ("loss_bo_mask", (pred_b, gt_b)), ("loss_invisible_mask", (pred_i, gt_i))]:
                    if pred is not None and gt is not None and pred.numel() > 0 and gt.numel() > 0:
                        if name == "loss_bo_mask":
                            valid_idx = torch.nonzero(torch.sum(gt.to(dtype=torch.float32), dim=(1,2))).squeeze(-1)
                            if len(valid_idx) > 0:
                                loss_val = F.binary_cross_entropy_with_logits(pred[valid_idx], gt[valid_idx], reduction="mean")
                            else:
                                loss_val = pred.sum() * 0.0
                        else:
                            loss_val = F.binary_cross_entropy_with_logits(pred, gt, reduction="mean")
                            
                        if getattr(self.mask_head, 'official_loss_weight', False) and name in ["loss_bo_mask", "loss_invisible_mask"]:
                            loss_val = loss_val * 0.25
                        losses[name] = loss_val
                    else:
                        losses[name] = torch.tensor(0.0, device=class_logits.device)
            else:
                # ?•ÁÑ°Ê≠?®£?¨Ô?Loss Ë®≠ÁÇ∫ 0
                losses.update({k: torch.tensor(0.0, device=class_logits.device) for k in ["loss_a_mask", "loss_vi_mask", "loss_bo_mask", "loss_invisible_mask"]})
            
        else: 
            # 4. Inference Logic
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i]})
            
            if self.mask_head is not None:
                mask_proposals = [p["boxes"] for p in result]
                if sum(p.numel() for p in mask_proposals) > 0:
                    mask_outputs = self.mask_head(features, mask_proposals, image_shapes)
                    boxes_per_image = [p.shape[0] for p in mask_proposals]
                    keys = ["amodal", "visible", "background_obj", "invisible"]
                    output_keys = ["pred_amodal_masks", "pred_visible_masks", "pred_background_masks", "pred_invisible_masks"]
                    
                    for k, out_k in zip(keys, output_keys):
                        logits = mask_outputs[k]
                        probs = logits.sigmoid()
                        probs_list = probs.split(boxes_per_image, 0)
                        
                        for i, r in enumerate(result):
                            if len(r["boxes"]) == 0:
                                r[out_k] = torch.empty((0, 1, 14, 14), device=class_logits.device)
                                continue
                            r[out_k] = probs_list[i]
                else:
                    for r in result:
                        for k in ["pred_amodal_masks", "pred_visible_masks", "pred_background_masks", "pred_invisible_masks"]:
                            r[k] = torch.empty((0, 1, 14, 14), device=class_logits.device)
        
        return result, losses

# ==============================================================================
# SECTION 4: TOP-LEVEL AMODAL SEGMENTATION MODEL
# ==============================================================================

class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_nodes, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        self.body = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

class SwinBackboneWithFPN(BackboneWithFPN):
    """Swin Transformer Ëº∏Âá∫??NHWCÔºåFPN ?ÄË¶?NCHWÔºåÈ??öÁ∂≠Â∫¶Ë???""
    def forward(self, x):
        x = self.body(x)
        for k, v in x.items():
            x[k] = v.permute(0, 3, 1, 2)
        x = self.fpn(x)
        return x

class AISFormerAmodal(nn.Module):
    def __init__(self, min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], 
                 backbone_name='resnet50', trainable_layers=3, pretrained_backbone=True, 
                 rpn_anchor_sizes=((32,), (64,), (128,), (256,), (512,)), rpn_aspect_ratios=((0.5, 1.0, 2.0),) * 5, 
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000, 
                 rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, 
                 num_classes=4, box_pooler_resolution=7, fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=256, 
                 positive_fraction=0.25, bbox_reg_weights=(10.0, 10.0, 5.0, 5.0), box_head_fc_dim=1024, 
                 roi_score_thresh=0.05, roi_nms_thresh=0.5, roi_detections_per_img=100, mask_pooler_resolution=14, 
                 frozen_stages=0, official_loss_weight=False, **kwargs):  # <--- [?∞Â?] frozen_stages ?ÉÊï∏
        super().__init__()
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        
        # ================= Backbone Selector =================
        print(f"??Ô∏? Initializing Backbone: {backbone_name}")
        weights = "DEFAULT" if pretrained_backbone else None
        
        # 1. ?™Â?‰øÆÊ≠£?ΩÂ? (?∏ÂÆπ??
        if '8_0gf' in backbone_name:
            print("   ?†Ô?  Auto-correcting model name: 'regnet_y_8_0gf' -> 'regnet_y_8gf'")
            backbone_name = backbone_name.replace('8_0gf', '8gf')
        if '3_2gf' in backbone_name:
            print("   ?†Ô?  Auto-correcting model name: 'regnet_y_3_2gf' -> 'regnet_y_3_2gf' (No change, just check)")
        

        # 2. ?óË©¶ËºâÂÖ•Ê®°Â?
        # try:
        #     norm_layer = FrozenBatchNorm2d if ('resnet' in backbone_name or 'regnet' in backbone_name) else None
        #     if norm_layer:
        #         base_backbone = get_model(backbone_name, weights=weights, norm_layer=norm_layer)
        #     else:
        #         base_backbone = get_model(backbone_name, weights=weights)
        # except ValueError as e:
        #     print(f"??Error loading model '{backbone_name}': {e}")
        #     print("   Please check available models using 'check_env.py'")
        #     raise e

        if 'convnextv2' in backbone_name:
            print(f"?ì¶ Loading {backbone_name} via timm (FCMAE Pre-trained)...")
            base_backbone = timm.create_model(
                backbone_name, 
                pretrained=pretrained_backbone, 
                features_only=True, 
                out_indices=(0, 1, 2, 3),
                drop_path_rate=0.5
            )
            # [‰øÆÊ≠£] ?πÁî®?πÊ??ºÂè´‰æÜÈ???Gradient CheckpointingÔºåÈÅø??TypeError
            try:
                base_backbone.set_grad_checkpointing(True)
                print("   ?ÑÔ?  Gradient Checkpointing enabled.")
            except AttributeError:
                print("   ?†Ô?  Gradient Checkpointing not supported by this timm version, skipping.")
        else:
            print(f"?ì¶ Loading {backbone_name} via torchvision...")
            try:
                norm_layer = FrozenBatchNorm2d if ('resnet' in backbone_name or 'regnet' in backbone_name) else None
                if norm_layer:
                    base_backbone = get_model(backbone_name, weights=weights, norm_layer=norm_layer)
                else:
                    base_backbone = get_model(backbone_name, weights=weights)
            except ValueError as e:
                print(f"??Error loading model '{backbone_name}': {e}")
                raise e

        # ======================================================================
        # [?∞Â?] Backbone Freezing Logic (È™®Âππ?çÁ??èËºØ)
        # ======================================================================
        if frozen_stages > 0:
            print(f"?ÑÔ?  Freezing Backbone up to stage {frozen_stages}")
            
            def set_eval(m):
                m.eval()

            if 'convnextv2' in backbone_name:
                for param in base_backbone.stem.parameters(): param.requires_grad = False
                base_backbone.stem.apply(set_eval)
                if frozen_stages >= 1:
                    for param in base_backbone.stages[0].parameters(): param.requires_grad = False
                    base_backbone.stages[0].apply(set_eval)
                if frozen_stages >= 2:
                    for param in base_backbone.stages[1].parameters(): param.requires_grad = False
                    base_backbone.stages[1].apply(set_eval)

            # --- ?ùÂ? Swin Transformer ?ÑÂ?ÁµêÈ?Ëº?---
            elif 'swin' in backbone_name or 'convnext' in backbone_name:
                # Stage 1 (?ÖÂê´ Stem/PatchEmbed ??Layer 1)
                if frozen_stages >= 1:
                    for param in base_backbone.features[0].parameters(): param.requires_grad = False
                    for param in base_backbone.features[1].parameters(): param.requires_grad = False
                    base_backbone.features[0].apply(set_eval)
                    base_backbone.features[1].apply(set_eval)
                
                # Stage 2 (?ÖÂê´ Downsample/PatchMerging ??Layer 2)
                if frozen_stages >= 2:
                    for param in base_backbone.features[2].parameters(): param.requires_grad = False
                    for param in base_backbone.features[3].parameters(): param.requires_grad = False
                    base_backbone.features[2].apply(set_eval)
                    base_backbone.features[3].apply(set_eval)

            # --- ?ùÂ? ResNet / RegNet ?ÑÂ?ÁµêÈ?Ëº?---
            elif 'resnet' in backbone_name or 'regnet' in backbone_name:
                # ?çÁ??Ä?ùÁ? Stem
                if hasattr(base_backbone, 'conv1'):
                    for param in base_backbone.conv1.parameters(): param.requires_grad = False
                if hasattr(base_backbone, 'bn1'):
                    for param in base_backbone.bn1.parameters(): param.requires_grad = False
                if hasattr(base_backbone, 'stem'):
                    for param in base_backbone.stem.parameters(): param.requires_grad = False

                # Freeze Stage 1 (Layer 1)
                if frozen_stages >= 1:
                    if hasattr(base_backbone, 'layer1'): # ResNet
                        for param in base_backbone.layer1.parameters(): param.requires_grad = False
                        base_backbone.layer1.apply(set_eval)
                    elif hasattr(base_backbone, 'trunk_output'): # RegNet
                        for param in base_backbone.trunk_output.block1.parameters(): param.requires_grad = False
                        base_backbone.trunk_output.block1.apply(set_eval)
                
                # Freeze Stage 2 (Layer 2)
                if frozen_stages >= 2:
                    if hasattr(base_backbone, 'layer2'): # ResNet
                        for param in base_backbone.layer2.parameters(): param.requires_grad = False
                        base_backbone.layer2.apply(set_eval)
                    elif hasattr(base_backbone, 'trunk_output'): # RegNet
                        for param in base_backbone.trunk_output.block2.parameters(): param.requires_grad = False
                        base_backbone.trunk_output.block2.apply(set_eval)

        # 3. ?™Â??§Êñ∑ FPN ÁØÄÈª?        if 'convnextv2' in backbone_name:
            # timm ?¥Êé•?ê‰??ÑÂ±§?öÈ??∏Ô?‰∏çÈ? Dummy Run
            in_channels_list = base_backbone.feature_info.channels()
            self.backbone = TimmBackboneWithFPN(base_backbone, in_channels_list, out_channels=256, extra_blocks=LastLevelMaxPool())
        else:
            if 'regnet' in backbone_name:
                return_nodes = {'trunk_output.block1': '0', 'trunk_output.block2': '1', 'trunk_output.block3': '2', 'trunk_output.block4': '3'}
                BackboneClass = BackboneWithFPN
            elif 'swin' in backbone_name:
                return_nodes = {'features.1': '0', 'features.3': '1', 'features.5': '2', 'features.7': '3'}
                BackboneClass = SwinBackboneWithFPN
            elif 'convnext' in backbone_name:  # <--- [?∞Â?] ConvNeXt ?ØÊè¥
                return_nodes = {'features.1': '0', 'features.3': '1', 'features.5': '2', 'features.7': '3'}
                BackboneClass = BackboneWithFPN # ConvNeXt Ëº∏Âá∫?ØÊ?Ê∫ñÁ? NCHWÔºå‰Ωø?®Â???BackboneWithFPN
            elif 'resnet' in backbone_name:
                return_nodes = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
                BackboneClass = BackboneWithFPN
            else:
                raise ValueError(f"Backbone {backbone_name} not supported (Only ResNet, RegNet, Swin).")

        # 4. Ë®àÁ??öÈ???(Dry Run)
            try:
                dummy = torch.randn(1, 3, 224, 224)
                feats = create_feature_extractor(base_backbone, return_nodes)(dummy)
                if 'swin' in backbone_name:
                    in_channels_list = [feats[str(i)].shape[-1] for i in range(4)]
                else:
                    in_channels_list = [feats[str(i)].shape[1] for i in range(4)]
            except Exception as e:
                print(f"??Error extracting features from {backbone_name}: {e}")
                raise e

        # 5. Âª∫Á? Backbone
            self.backbone = BackboneClass(base_backbone, return_nodes, in_channels_list, out_channels=256, extra_blocks=LastLevelMaxPool())

        out_channels = self.backbone.out_channels

        # 6. RPN & ROI Heads Setup
        anchor_generator = AnchorGenerator(sizes=rpn_anchor_sizes, aspect_ratios=rpn_aspect_ratios)
        rpn_head = RPNHead(out_channels, anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(anchor_generator, rpn_head, rpn_fg_iou_thresh, rpn_bg_iou_thresh, rpn_batch_size_per_image, rpn_positive_fraction, rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        box_roi_pool = AlignedMultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=box_pooler_resolution, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = box_head_fc_dim
        box_head = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(out_channels * resolution ** 2, representation_size), nn.ReLU(), nn.Linear(representation_size, representation_size), nn.ReLU())
        
        box_predictor = FastRCNNCustomPredictor(representation_size, num_classes)
        
        mask_head = AISFormerHead(out_channels, num_classes, mask_pooler_resolution, official_loss_weight=official_loss_weight, **kwargs)
        
        self.roi_heads = AISFormerRoIHeads(box_roi_pool, box_head, box_predictor, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights, mask_head=mask_head, score_thresh=roi_score_thresh, nms_thresh=roi_nms_thresh, detections_per_img=roi_detections_per_img, mask_rasterization_chunk_size=64)

    def forward(self, images, targets=None):
        if self.training and targets is None: raise ValueError("In training mode, targets should be passed")
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]; assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor): features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        
        if not self.training:
            for i, (pred, im_s, o_im_s) in enumerate(zip(detections, images.image_sizes, original_image_sizes)):
                for k in ["pred_amodal_masks", "pred_visible_masks", "pred_background_masks", "pred_invisible_masks"]:
                    if k in pred:
                        mask_prob = pred[k]
                        boxes = pred["boxes"]
                        if mask_prob.numel() == 0:
                            pred[k] = torch.empty((0, 1, o_im_s[0], o_im_s[1]), device=mask_prob.device)
                            continue
                        
                        # [Optimization] Move to CPU for pasting to avoid GPU OOM
                        # Since pasting large masks (e.g. 4000x3000) can take several GBs
                        device = mask_prob.device
                        mask_prob_cpu = mask_prob.cpu()
                        boxes_cpu = boxes.cpu()
                        
                        pasted_mask = paste_masks_in_image(mask_prob_cpu, boxes_cpu, o_im_s)
                        pred[k] = pasted_mask.to(device)
        
        losses = {}; losses.update(detector_losses); losses.update(proposal_losses)
        if self.training: return losses
        return detections

