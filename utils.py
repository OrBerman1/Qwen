import torch
from dataset import CocoDetection
from coco_eval import CocoEvaluator
import numpy as np


def convert_to_xywh(boxes):
    boxes = [[xmin, ymin, xmax - xmin, ymax - ymin] for xmin, ymin, xmax, ymax in boxes]
    return torch.tensor(boxes)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction["boxes"]) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"]
        labels = prediction["labels"]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def get_class_ap(coco_eval, num_classes):
    ap_per_category = {}
    precisions = coco_eval.eval["precision"]
    for i in range(num_classes):
        precision = precisions[:, :, i, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        ap_per_category[i + 1] = ap * 100
    return ap_per_category


def get_class_ar(coco_eval, num_classes):
    ar_per_category = {}
    recalls = coco_eval.eval["recall"]
    for i in range(num_classes):
        recall = recalls[:, i, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        ar_per_category[i + 1] = ar * 100
    return ar_per_category


def evaluate(outputs, dl, num_classes=2):
    evaluator = CocoEvaluator(coco_gt=dl.dataset.coco, iou_types=["bbox"])
    evaluator.update(outputs)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    # Create the COCOeval object
    coco_eval = evaluator.coco_eval["bbox"]

    # Optionally, you can evaluate the result using coco_eval
    ar_per_category = get_class_ar(coco_eval, num_classes)
    ap_per_category = get_class_ap(coco_eval, num_classes)
    print(f'Average Precision (AP) per category: {ap_per_category}')
    print(f'Average Recall (AR) per category: {ar_per_category}')
