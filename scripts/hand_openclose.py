import os
import sys
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOD_ROOT = os.path.join(PROJECT_ROOT, "hand_object_detector")
HOD_LIB = os.path.join(HOD_ROOT, "lib")
sys.path.insert(0, HOD_ROOT)
sys.path.insert(0, HOD_LIB)

from model.utils.config import cfg, cfg_from_file
from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from model.roi_layers import nms
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg_from_list

def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im_r = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                           interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im_r)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def load_hand_detector(
    model_path: str,
    cfg_file: str,
    device: str = "cuda",
):
    cfg_from_file(cfg_file)
    cfg.USE_GPU_NMS = (device == "cuda")
    cfg.CUDA = (device == "cuda")

    set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']
    cfg_from_list(set_cfgs)

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    model = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
    model.create_architecture()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    model.to(device)
    model.eval()
    return model, pascal_classes


def detect_hand_contact_state(model, im_bgr, device="cuda", thresh_hand=0.5):
    """
    Returns contact state for detected hands.
    Contact states: 0=N(no contact), 1=S(self), 2=O(other person),
                    3=P(portable object), 4=F(stationary/furniture)
    Gripper closed if contact_state in {3, 4} (P or F).
    """
    im_data = torch.FloatTensor(1).cuda()
    im_info = torch.FloatTensor(1).cuda()
    num_boxes = torch.LongTensor(1).cuda()
    gt_boxes = torch.FloatTensor(1).cuda()
    box_info = torch.FloatTensor(1)
    blobs, im_scales = _get_image_blob(im_bgr)
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)
    im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.resize_(1, 1, 5).zero_()
    num_boxes.resize_(1).zero_()
    box_info.resize_(1, 1, 5).zero_() 
    with torch.no_grad():
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = model(im_data, im_info, gt_boxes, num_boxes, box_info)
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    contact_vector = loss_list[0][0]
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()
    box_deltas = bbox_pred.data
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        num_classes_x4 = box_deltas.size(-1)
        if device == "cuda":
            box_deltas = box_deltas.reshape(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        else:
            box_deltas = box_deltas.reshape(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        box_deltas = box_deltas.reshape(1, -1, num_classes_x4)
    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    # Check hand detections (class index 2 = 'hand')
    hand_class_idx = 2
    inds = torch.nonzero(scores[:, hand_class_idx] > thresh_hand).reshape(-1)

    if inds.numel() == 0:
        return None  # No hand detected

    cls_scores = scores[:, hand_class_idx][inds]
    cls_boxes = pred_boxes[inds, :]
    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds]), 1)

    _, order = torch.sort(cls_scores, 0, True)
    cls_dets = cls_dets[order]
    keep = nms(cls_dets[:, :4], cls_dets[:, 4], cfg.TEST.NMS)
    cls_dets = cls_dets[keep.reshape(-1).long()]
    # Take highest confidence detection
    best = cls_dets[0]
    contact_state = int(best[5].item())
    return contact_state


def detect_hand_states_for_video(
    npz_path: str,
    output_path: str,
    model_path: str,
    cfg_file: str,
    device: str = "auto",
    thresh_hand: float = 0.5,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_hand_detector(model_path, cfg_file, device=device)

    data = np.load(npz_path, allow_pickle=True)
    rgb_frames = data["rgb"]  # (N, H, W, 3)
    num_frames = len(rgb_frames)
    print(f"     Processing {num_frames} frames for hand state detection")

    # Per-frame: True = open, False = closed
    hand_open_states = np.ones(num_frames, dtype=np.bool_)

    for i in range(num_frames):
        # Convert RGB to BGR for the detector
        im_bgr = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2BGR)
        contact_state = detect_hand_contact_state(model, im_bgr, device=device, thresh_hand=thresh_hand)
        if contact_state is not None:
            # P(3) or F(4) = closed/gripping; else = open
            hand_open_states[i] = contact_state not in (3, 4)
        # If no hand detected, default to open

        if (i + 1) % 25 == 0:
            print(f"     [{i+1}/{num_frames}]")

    np.save(output_path, hand_open_states)
    print(f"     Saved hand states ({num_frames} frames) to {output_path}")
    return output_path