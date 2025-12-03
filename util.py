import cv2
import numpy as np

def preprocess(frame, input_size=(640, 640)):
    """Resize and normalize a frame for TFLite YOLO model."""
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, input_size)
    normalized = resized / 255.0
    return normalized.astype(np.float32), (h, w)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def parse_yolo_output(output, conf_threshold=0.3):
    """
    Convert YOLO raw output (1, 30, 8400) into usable detections.
    Returns [x1, y1, x2, y2, conf, class_id]
    """

    output = np.squeeze(output)  # (30, 8400)

    boxes = output[:4, :]
    scores = output[4:5, :]
    class_probs = output[5:, :]

    conf = sigmoid(scores)
    class_probs = sigmoid(class_probs)

    final_scores = conf * class_probs.max(axis=0)
    class_ids = class_probs.argmax(axis=0)

    mask = final_scores > conf_threshold
    final_scores = final_scores[mask]
    class_ids = class_ids[mask]
    boxes = boxes[:, mask]

    detections = []
    for i in range(boxes.shape[1]):
        x, y, w, h = boxes[:, i]
        detections.append([x, y, x + w, y + h, float(final_scores[i]), int(class_ids[i])])

    return detections

def nms(detections, iou_threshold=0.45):
    """Apply Non-Max Suppression."""
    if len(detections) == 0:
        return []

    dets = np.array(detections)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_j = (x2[idxs[1:]] - x1[idxs[1:]]) * (y2[idxs[1:]] - y1[idxs[1:]])
        union = area_i + area_j - inter_area

        iou = inter_area / union
        idxs = idxs[1:][iou < iou_threshold]

    return dets[keep].tolist()

def load_class_names(path="data/classes.txt"):
    with open(path, "r") as f:
        return [c.strip() for c in f.readlines()]
