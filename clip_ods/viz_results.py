import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_precision_recall(metrics):
    """
    Compute precision and recall per class.

    Args:
        metrics (dict): Dictionary containing TP, FP, FN per class.

    Returns:
        dict: Dictionary with precision and recall per class.
    """
    precision_recall = {}
    for class_name, counts in metrics.items():
        TP = counts['TP']
        FP = counts['FP']
        FN = counts['FN']
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        precision_recall[class_name] = {
            'precision': precision,
            'recall': recall
        }
    return precision_recall

def plot_precision_recall(precision_recall):
    """
    Plot precision and recall per class.
    """
    classes = list(precision_recall.keys())
    precisions = [precision_recall[cls]['precision'] for cls in classes]
    recalls = [precision_recall[cls]['recall'] for cls in classes]

    x = range(len(classes))

    plt.figure(figsize=(20, 8))
    plt.bar(x, precisions, width=0.4, label='Precision', align='center')
    plt.bar(x, recalls, width=0.4, label='Recall', align='edge')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Precision and Recall per Class')
    plt.xticks(x, classes, rotation='vertical')
    plt.legend()
    plt.tight_layout()
    plt.savefig('precision_recall_per_class.png')
    plt.close()

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea


def evaluate_detections(annotations, detection_results, categories, iou_threshold=0.5):
    metrics = {class_name: {'TP': 0, 'FP': 0, 'FN': 0} for class_name in categories.values()}
    for annotation, detection_result in zip(annotations, detection_results):
        gt_boxes = annotation.get('boxes', [])
        gt_labels = annotation.get('labels', [])
        detections_by_category = detection_result['detections_by_category']
        gt_boxes_by_class = {}
        for box, label in zip(gt_boxes, gt_labels):
            class_name = categories.get(label)
            if class_name:
                gt_boxes_by_class.setdefault(class_name, []).append(box)
        for class_name in categories.values():
            gt_boxes_class = gt_boxes_by_class.get(class_name, [])
            detections = detections_by_category.get(class_name, {}).get('detections', {})
            pred_boxes = detections.get('boxes', [])
            matched_gt = []
            for pred_box in pred_boxes:
                match_found = False
                for idx, gt_box in enumerate(gt_boxes_class):
                    if idx in matched_gt:
                        continue
                    if compute_iou(pred_box, gt_box) >= iou_threshold:
                        metrics[class_name]['TP'] += 1
                        matched_gt.append(idx)
                        match_found = True
                        break
                if not match_found:
                    metrics[class_name]['FP'] += 1
            metrics[class_name]['FN'] += len(gt_boxes_class) - len(matched_gt)
    return metrics


def plot_iou_distribution(annotations, detection_results):
    ious = []
    for annotation, detection_result in zip(annotations, detection_results):
        gt_boxes = annotation.get('boxes', [])
        detections_by_category = detection_result['detections_by_category']
        for class_name, data in detections_by_category.items():
            detections = data.get('detections', {})
            pred_boxes = detections.get('boxes', [])
            for pred_box in pred_boxes:
                for gt_box in gt_boxes:
                    ious.append(compute_iou(pred_box, gt_box))
    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=50, color='orange', alpha=0.7)
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.title('IoU Distribution of Matched Boxes')
    plt.tight_layout()
    plt.savefig('iou_distribution.png')
    plt.close()


def plot_confusion_matrix(annotations, detection_results, categories):
    """
    Plot a confusion matrix for detected and ground-truth labels.
    """
    gt_labels = []
    pred_labels = []

    for annotation, detection_result in zip(annotations, detection_results):
        # Append ground truth labels
        gt_labels.extend(annotation.get('labels', []))

        detections_by_category = detection_result['detections_by_category']
        for class_name, data in detections_by_category.items():
            if class_name not in categories.values():
                continue  # Skip if class_name is not in categories
            detections = data.get('detections', {})
            pred_labels.extend(
                [list(categories.keys())[list(categories.values()).index(class_name)]] * len(detections.get('boxes', []))
            )

    # Map labels to names
    gt_label_names = [categories[label] for label in gt_labels if label in categories]
    pred_label_names = [categories[label] for label in pred_labels if label in categories]

    # Ensure both lists have the same length
    min_len = min(len(gt_label_names), len(pred_label_names))
    gt_label_names = gt_label_names[:min_len]
    pred_label_names = pred_label_names[:min_len]

    # Get unique classes
    all_classes = sorted(list(set(gt_label_names + pred_label_names)))

    # Generate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt_label_names, pred_label_names, labels=all_classes)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_classes, yticklabels=all_classes, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()



def plot_class_distribution(annotations, detection_results, categories):
    """
    Plot the actual and detected class distributions as pie charts.
    """
    # Initialize counts for ground truth and predicted classes
    gt_counts = {class_name: 0 for class_name in categories.values()}
    pred_counts = {class_name: 0 for class_name in categories.values()}

    # Count ground truth classes
    for annotation in annotations:
        gt_labels = annotation.get('labels', [])
        for label in gt_labels:
            class_name = categories.get(label)
            if class_name:
                gt_counts[class_name] += 1

    # Count predicted classes
    for detection_result in detection_results:
        detections_by_category = detection_result['detections_by_category']
        for class_name, data in detections_by_category.items():
            if class_name not in pred_counts:
                continue  # Skip if class_name is not in categories
            detections = data.get('detections', {})
            pred_counts[class_name] += len(detections.get('boxes', []))

    # Filter out classes with zero counts for better visualization
    gt_counts = {k: v for k, v in gt_counts.items() if v > 0}
    pred_counts = {k: v for k, v in pred_counts.items() if v > 0}

    # Prepare data for plotting
    gt_labels = list(gt_counts.keys())
    gt_values = list(gt_counts.values())
    pred_labels = list(pred_counts.keys())
    pred_values = list(pred_counts.values())

    # Plot ground truth distribution
    plt.figure(figsize=(12, 8))
    plt.pie(
        gt_values,
        labels=gt_labels,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10},
    )
    plt.title('Ground Truth Class Distribution')
    plt.tight_layout()
    plt.savefig('ground_truth_distribution.png')
    plt.close()

    # Plot predicted distribution
    plt.figure(figsize=(12, 8))
    plt.pie(
        pred_values,
        labels=pred_labels,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10},
    )
    plt.title('Predicted Class Distribution')
    plt.tight_layout()
    plt.savefig('predicted_distribution.png')
    plt.close()




def plot_classwise_ap(metrics):
    class_aps = {class_name: counts['TP'] / (counts['TP'] + counts['FP']) if counts['TP'] + counts['FP'] > 0 else 0
                 for class_name, counts in metrics.items()}
    plt.figure(figsize=(12, 6))
    plt.bar(class_aps.keys(), class_aps.values(), color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Average Precision (AP)')
    plt.title('Per-Class Average Precision')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('per_class_ap.png')
    plt.close()


def main():
    with open('/storage/ice1/9/3/kkundurthy3/synthetic_eval_bb/synthetic_annotations_bb.json', 'r') as f:
        annotations = json.load(f)
    with open('/storage/ice1/9/3/kkundurthy3/synthetic_eval_bb/detection_eval_result_4000.json', 'r') as f:
        detection_results = json.load(f)

    categories = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    }

    metrics = evaluate_detections(annotations, detection_results, categories)

    # Generate visualizations
    plot_precision_recall(precision_recall=compute_precision_recall(metrics))
    plot_iou_distribution(annotations, detection_results)
    plot_confusion_matrix(annotations, detection_results, categories)
    plot_class_distribution(annotations, detection_results, categories)
    plot_classwise_ap(metrics)


if __name__ == "__main__":
    main()
