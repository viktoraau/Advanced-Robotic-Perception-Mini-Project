
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import csv

import cv2

from traditional_detector_codex import (
    find_ball_candidates,
    BallTracker
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = PROJECT_DIR / "denmark_wales_clip_1min.mp4"
ANNOTATION_ZIP = PROJECT_DIR / "evaluation_annotations.zip"

OUTPUT_DIR = PROJECT_DIR / "traditional_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

FRAME_STEP = 5
IOU_THRESHOLD = 0.50


# --------------------------------------------------
# LOAD CVAT ANNOTATIONS
# --------------------------------------------------

def load_annotations(zip_path):

    with zipfile.ZipFile(zip_path) as archive:

        xml_name = next(
            name for name in archive.namelist()
            if name.endswith("annotations.xml")
        )

        root = ET.fromstring(archive.read(xml_name))

    annotations = {}

    for track in root.findall("track"):

        if track.attrib.get("label") != "ball":
            continue

        for box in track.findall("box"):

            if box.attrib.get("outside") == "1":
                continue

            frame = int(box.attrib["frame"])

            x1 = float(box.attrib["xtl"])
            y1 = float(box.attrib["ytl"])
            x2 = float(box.attrib["xbr"])
            y2 = float(box.attrib["ybr"])

            annotations.setdefault(frame, []).append(
                [x1, y1, x2, y2]
            )

    return annotations


# --------------------------------------------------
# INTERSECTION OVER UNION
# --------------------------------------------------

def calculate_iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def main():

    print("Loading ground truth...")
    annotations = load_annotations(ANNOTATION_ZIP)

    print(f"Annotated frames: {len(annotations)}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = BallTracker()

    TP = 0
    FP = 0
    FN = 0

    frame_number = 0
    evaluated_frames = 0

    csv_path = OUTPUT_DIR / "traditional_predictions.csv"

    with open(csv_path, "w", newline="") as csvfile:

        csv_writer = csv.writer(csvfile)

        csv_writer.writerow([
            "frame",
            "detected",
            "pred_x1",
            "pred_y1",
            "pred_x2",
            "pred_y2",
            "ground_truth_present",
            "IoU"
        ])

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Run detector and tracker on EVERY frame
            _, candidates = find_ball_candidates(frame)

            box, center, detected = tracker.update(candidates)

            # Only evaluate the annotated frames
            if frame_number % FRAME_STEP == 0:

                evaluated_frames += 1

                ground_truth = annotations.get(frame_number, [])

                prediction = None

                if detected and box is not None:

                    x, y, w, h = box

                    prediction = [
                        float(x),
                        float(y),
                        float(x + w),
                        float(y + h)
                    ]

                iou = 0.0

                # One football, one prediction
                if prediction is not None and ground_truth:

                    iou = max(
                        calculate_iou(prediction, gt)
                        for gt in ground_truth
                    )

                    if iou >= IOU_THRESHOLD:
                        TP += 1
                    else:
                        FP += 1
                        FN += 1

                elif prediction is not None:

                    FP += 1

                elif ground_truth:

                    FN += 1

                csv_writer.writerow([
                frame_number,
                prediction is not None,
                *(prediction if prediction else ["", "", "", ""]),
                bool(ground_truth),
                round(iou, 4)
            ])

            frame_number += 1

    cap.release()

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\n========== TRADITIONAL CV RESULTS ==========")

    print(f"Evaluated frames: {evaluated_frames}")
    print(f"True Positives:   {TP}")
    print(f"False Positives:  {FP}")
    print(f"False Negatives:  {FN}")

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    print("============================================")

    print(f"\nFrame predictions saved to: {csv_path}")


if __name__ == "__main__":
    main()
