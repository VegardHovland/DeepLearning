import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    #Extract x and y coordinates 
    x_min = max(gt_box[0] , prediction_box[0])
    y_min = max(gt_box[1] , prediction_box[1])
    x_max = min(gt_box[2] , prediction_box[2])
    y_max = min(gt_box[3] , prediction_box[3])

    x_int = x_max - x_min
    y_int = y_max - y_min

    # Compute intersection
    intersect = np.maximum(0, x_int) * np.maximum(0, y_int)
    intersect = np.abs(intersect)

    # Compute box area
    gtArea = (gt_box[2] - gt_box[0]) * (gt_box[3] -gt_box[1])
 
    predArea = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])

    # Compute union
    iou = intersect / (gtArea + predArea - intersect)
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    return num_tp / (num_tp + num_fn)



def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    # Feilen min ligger i denne koden! Den passer testen i test filen, men fukker opp i det store bildet
    pred_matches = []
    gt_matches = []
    for gt in gt_boxes:
        best_iou = 0
        matching_thruth = None 
        for pred in prediction_boxes:
            iou = calculate_iou(pred, gt)
            if iou >=  iou_threshold:
                if iou > best_iou:                                     # Find best match, if equal first detected counts!
                    best_iou = iou
                    matching_thruth = gt
        if best_iou >= iou_threshold and matching_thruth is not None:    # Dont think last statement is necessary
            pred_matches.append(pred)
            gt_matches.append(matching_thruth)
    return np.array(pred_matches), np.array(gt_matches)         # Returns predicion vector and truth vector both sorted after decreasing IoU


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    num_true = len(gt_boxes) #.shape[0]
    num_pred = len(prediction_boxes) #.shape[0]

    pred_match, truth_match = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    tp = pred_match.shape[0]
    fp = num_true - tp
    fn = num_pred - tp
    results = {}
    results["true_pos"] = tp
    results["false_pos"] = fp
    results["false_neg"] = fn

    return results


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    tp = 0
    fp = 0 
    fn = 0
    # Count over all images
    for i in range(len(all_prediction_boxes)):
        res = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        tp += res["true_pos"]
        fp += res["false_pos"] 
        fn += res["false_neg"] 

    return (calculate_precision(tp, fp, fn), calculate_recall(tp, fp, fn))


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = [] 
    recalls = []
    for ct in confidence_thresholds:
        conf_pred_boxes = [] 
        #Find prediction boxes over IoU threshhold
        for i in range(len(confidence_scores)):
            conf_pred_boxes.append(all_prediction_boxes[i][confidence_scores[i] >= ct, :])
        #Calculate precicion and recall
        precision, recall = calculate_precision_recall_all_images(conf_pred_boxes, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
    #print(np.array(precisions))
    #print(recalls)
    # THIS IS CORRECT TESTED VS Friends Code! Fault comes from something else....----------
    return np.array(precisions), np.array(recalls)

def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)

    # YOUR CODE HERE

    average_precision = []
    for rl in recall_levels:
        p_interp = []
        for r,p in zip(precisions, recalls):
            if r >= rl:
                p_interp.append(p)
            else:
                p_interp.append(0)
        if p_interp:
            average_precision.append(max(p_interp))
        else:
            average_precision.append(0)
        
    return np.mean(np.array(average_precision))


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
