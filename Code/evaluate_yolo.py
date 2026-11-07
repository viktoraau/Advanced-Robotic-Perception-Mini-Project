
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = PROJECT_DIR / "denmark_wales_clip_1min.mp4"
ANNOTATION_ZIP = PROJECT_DIR / "evaluation_annotations.zip"

MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "soccer_ball_yolov8n"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.50

# CVAT frame step used when creating the task
FRAME_STEP = 5


# --------------------------------------------------
# 2. LOAD CVAT ANNOTATIONS
# --------------------------------------------------


def load_annotations(zip_path):
    """
    Reads CVAT for video 1.1 annotations.

    Returns:
        Dictionary:
        original video frame number -> list of bounding boxes
    """

    with zipfile.ZipFile(zip_path, "r") as archive:

        xml_name = next(
            name for name in archive.namelist()
            if name.endswith("annotations.xml")
        )

        xml_data = archive.read(xml_name)

    root = ET.fromstring(xml_data)

    annotations = {}

    # CVAT for video stores annotations inside tracks
    for track in root.findall("track"):

        if track.attrib.get("label") != "ball":
            continue

        for box in track.findall("box"):

            # Ignore boxes where the object is outside
            if box.attrib.get("outside") == "1":
                continue

            frame_number = int(box.attrib["frame"])

            x1 = float(box.attrib["xtl"])
            y1 = float(box.attrib["ytl"])
            x2 = float(box.attrib["xbr"])
            y2 = float(box.attrib["ybr"])

            annotations.setdefault(frame_number, []).append(
                [x1, y1, x2, y2]
            )

    return annotations



# --------------------------------------------------
# 3. CALCULATE INTERSECTION OVER UNION
# --------------------------------------------------

def calculate_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (
        (box_a[2] - box_a[0])
        * (box_a[3] - box_a[1])
    )

    area_b = (
        (box_b[2] - box_b[0])
        * (box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# --------------------------------------------------
# 4. MATCH PREDICTIONS TO GROUND TRUTH
# --------------------------------------------------

def match_boxes(predictions, ground_truth):

    matches = []

    used_gt = set()

    # Highest confidence predictions are considered first
    predictions = sorted(
        predictions,
        key=lambda box: box[4],
        reverse=True
    )

    for prediction in predictions:

        pred_box = prediction[:4]

        best_iou = 0
        best_index = None

        for index, gt_box in enumerate(ground_truth):

            if index in used_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_iou >= IOU_THRESHOLD and best_index is not None:

            matches.append((prediction, ground_truth[best_index]))
            used_gt.add(best_index)

    return matches, len(used_gt)


# --------------------------------------------------
# 5. MAIN EVALUATION
# --------------------------------------------------

def main():

    print("Loading annotations...")
    annotations = load_annotations(ANNOTATION_ZIP)

    print(f"Annotated frames: {len(annotations)}")

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total video frames: {total_frames}")

    TP = 0
    FP = 0
    FN = 0

    output_video_path = OUTPUT_DIR / "yolo_evaluation.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / FRAME_STEP,
        (width, height)
    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Evaluate only the sampled frames
        if frame_number % FRAME_STEP == 0:

            results = model.predict(
                frame,
                imgsz=1280,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            predictions = []

            for box in results[0].boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                predictions.append([
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    confidence
                ])

            ground_truth = annotations.get(frame_number, [])

            matches, matched_gt = match_boxes(
                predictions,
                ground_truth
            )

            TP += len(matches)
            FP += len(predictions) - len(matches)
            FN += len(ground_truth) - matched_gt

            # Draw ground truth in green
            for gt in ground_truth:

                x1, y1, x2, y2 = map(int, gt)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            # Draw predictions in red
            for pred in predictions:

                x1, y1, x2, y2 = map(int, pred[:4])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"{pred[4]:.2f}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1
                )

            writer.write(frame)

        frame_number += 1

    cap.release()
    writer.release()

    # --------------------------------------------------
    # 6. CALCULATE METRICS
    # --------------------------------------------------

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    print("\n========== EVALUATION RESULTS ==========")

    print(f"True Positives:  {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    print("========================================")

    print(f"\nVisualisation saved to: {output_video_path}")


if __name__ == "__main__":
    main()
