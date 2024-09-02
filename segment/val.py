import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.general import (
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import box_iou
from utils.plots import output_to_target, plot_val_study
from utils.segment.dataloaders import create_dataloader
from utils.segment.general import mask_iou, process_mask, process_mask_native, scale_image
from utils.segment.metrics import Metrics, ap_per_class_box_and_mask
from utils.segment.plots import plot_images_and_masks
from utils.torch_utils import de_parallel, select_device, smart_inference_mode

def save_one_txt(predn, save_conf, shape, file):
    """Saves detection results in txt format; includes class, xywh (normalized), optionally confidence if `save_conf` is
    True.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map, pred_masks):
    """
    Saves a JSON file with detection results including bounding boxes, category IDs, scores, and segmentation masks.

    Example JSON result: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}.
    """
    from pycocotools.mask import encode

    def single_encode(x):
        """Encodes binary mask arrays into RLE (Run-Length Encoding) format for JSON serialization."""
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
                "segmentation": rles[i],
            }
        )


def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels.
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val-seg",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    overlap=False,
    mask_downsample_ratio=1,
    compute_loss=None,
    callbacks=Callbacks(),
):
    """Validates a YOLOv5 segmentation model on specified dataset, producing metrics, plots, and optional JSON
    output.
    """
    if save_json:
        check_requirements("pycocotools>=2.0.6")
        process = process_mask_native  # more accurate
    else:
        process = process_mask  # faster

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)  # load YOLOv5 model
    else:
        model, pt, jit, engine = model, False, False, False
        device = select_device(device, batch_size=batch_size, workers=workers)  # use same device as data loaders
        half &= device.type != "cpu"  # half precision only supported on CUDA

    # Load dataset
    dataset = create_dataloader(data, imgsz, batch_size, workers, prefix=colorstr("val: "), hyp=None, augment=augment)[0]
    names = dataset.dataset.names  # class names
    if save_json:
        json_file = increment_path(Path(project) / name / "results.json", exist_ok=exist_ok)
        with open(json_file, "w") as f:
            json.dump([], f)
        json_file = Path(json_file)
        LOGGER.info(f"Saving results to {json_file}")

    # Start validation
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    callbacks.on_fit_start()
    results = []
    iou_stats = []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataset, desc="Validation", bar_format=TQDM_BAR_FORMAT)):
        batch_i = dataset.batch_sampler.batch_index[0] if dataset.batch_sampler is not None else batch_i  # batch index
        if training:
            if model.pt:  # PyTorch model
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # 0 - 1
            else:
                imgs = imgs.to(device, non_blocking=True).float()
            if half:
                imgs = imgs.half()
            with torch.no_grad():
                preds = model(imgs, augment=augment)
        else:
            if half:
                imgs = imgs.half()
            with torch.no_grad():
                preds = model(imgs)
        if model.pt:
            # PyTorch model
            pred = preds
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=True, max_det=max_det)
        else:
            # TensorRT or ONNX model
            pred = preds[0] if isinstance(preds, tuple) else preds
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=True, max_det=max_det)

        if save_txt or save_json:
            pred = [p.cpu() for p in pred]  # move predictions to CPU
        if save_txt:
            for si, (p, s) in enumerate(zip(pred, shapes)):
                save_one_txt(p, save_conf, s, save_dir / f"{Path(paths[si]).stem}.txt")
        if save_json:
            with open(json_file, "r") as f:
                jdict = json.load(f)
            for si, (p, s, path, m) in enumerate(zip(pred, shapes, paths, preds)):
                save_one_json(p, jdict, Path(path), dataset.dataset.class_map, m)
            with open(json_file, "w") as f:
                json.dump(jdict, f)

        # Compute IoU metrics
        targets = targets.tolist()
        preds = [p.tolist() for p in pred]
        for p, s, shape in zip(preds, targets, shapes):
            pred_masks = np.array([np.array(p)[:, :4] * np.array(s[1:]).tolist() for p in preds])
            gt_masks = np.array([np.array(t)[:, :4] * np.array(s[1:]).tolist() for t in targets])
            iou = process_batch(p, gt_masks, iou_thres, pred_masks=pred_masks, gt_masks=gt_masks, overlap=overlap, masks=True)
            iou_stats.append(iou.mean(0).cpu().numpy())

    # Calculate average IoU
    iou_stats = np.mean(np.array(iou_stats), axis=0)
    iou_thres = np.arange(0.5, 1.0, 0.05)

    # Plot IoU metrics
    plt.figure(figsize=(10, 6))
    plt.plot(iou_thres, iou_stats, marker='o', linestyle='-', color='b')
    plt.xlabel('IoU Threshold')
    plt.ylabel('Average IoU')
    plt.title('IoU Metrics Plot')
    plt.grid(True)
    plt.show()

    # Finish
    callbacks.on_fit_end()
    if plots:
        plot_images_and_masks(preds, targets, names, save_dir=save_dir)
    LOGGER.info("Validation complete.")
    return results, iou_stats
