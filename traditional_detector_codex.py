"""Small-football detection with classical OpenCV image processing.

The detector looks for compact, nearly circular bright regions, then follows
the most plausible candidate from frame to frame. It is intended for the
included 640x360 broadcast clip; the ball is only a few pixels across.

Examples:
    python traditional_detector_codex.py
    python traditional_detector_codex.py --video another_clip.mp4 --output tracked.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO = Path(__file__).with_name("denmark_wales_clip_1min.mp4")


def find_ball_candidates(frame):
    """Return compact bright-object candidates as (x, y, w, h, score)."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # The ball is white, but grass, pale advertising and shirts also pass this
    # threshold. Shape, size, and tracking below remove many of those regions.
    bright = cv2.inRange(hsv, np.array((0, 0, 135)), np.array((180, 150, 255)))

    # Skip the score graphic and audience while keeping the whole playable area.
    field_roi = np.zeros_like(bright)
    polygon = np.array([
        (0, int(height * 0.25)),
        (width, int(height * 0.25)),
        (width, height),
        (0, height),
    ], dtype=np.int32)
    cv2.fillPoly(field_roi, [polygon], 255)
    bright = cv2.bitwise_and(bright, field_roi)

    # Do not open/erode this mask: at 640x360 the ball is often only 3-6 px,
    # so a morphology operation can erase it completely.
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if not (2 <= w <= 22 and 2 <= h <= 22 and 2 <= area <= 220):
            continue

        aspect = w / float(h)
        if not 0.55 <= aspect <= 1.8:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.10:
            continue

        # Prefer round, small candidates with a high proportion of white pixels.
        roi = bright[y:y + h, x:x + w]
        white_fraction = cv2.countNonZero(roi) / float(w * h)
        # A football-sized white blob should be surrounded by grass. This
        # rejects similarly sized white details in the crowd and score bug.
        pad = 5
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        neighborhood = hsv[y0:y1, x0:x1]
        green = (
            (neighborhood[:, :, 0] >= 22)
            & (neighborhood[:, :, 0] <= 105)
            & (neighborhood[:, :, 1] >= 25)
        )
        # Exclude the candidate itself so white pixels cannot inflate the
        # surrounding-grass measurement.
        green[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
        grass_fraction = float(np.mean(green))
        if grass_fraction < 0.08:
            continue

        size_penalty = abs(max(w, h) - 6.0) / 18.0
        score = (
            1.0 * circularity
            + 0.45 * white_fraction
            + 0.7 * grass_fraction
            - size_penalty
        )
        candidates.append((x, y, w, h, score))

    return bright, candidates


class BallTracker:
    """Greedy constant-velocity tracker for the frame-by-frame candidates."""

    def __init__(self, max_missed=5, max_distance=48.0):
        self.center = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.box = None
        self.missed = 0
        self.max_missed = max_missed
        self.max_distance = max_distance

    def update(self, candidates):
        prediction = None
        if self.center is not None:
            prediction = self.center + self.velocity

        if not candidates:
            return self._miss(prediction)

        if prediction is None:
            # Start immediately from a compact candidate on the pitch. The
            # short confirmation logic tried before caused delayed, unstable
            # acquisition and is intentionally avoided here.
            pitch_candidates = [item for item in candidates if item[1] >= 95]
            chosen = max(pitch_candidates or candidates, key=lambda item: item[4])
        else:
            ranked = []
            for item in candidates:
                x, y, w, h, score = item
                point = np.array((x + w / 2.0, y + h / 2.0), dtype=np.float32)
                distance = float(np.linalg.norm(point - prediction))
                if distance <= self.max_distance:
                    ranked.append((distance - 5.0 * score, item, point))
            if not ranked:
                return self._miss(prediction)
            _, chosen, point = min(ranked, key=lambda row: row[0])

        x, y, w, h, _ = chosen
        new_center = np.array((x + w / 2.0, y + h / 2.0), dtype=np.float32)
        if self.center is not None:
            measured_velocity = new_center - self.center
            self.velocity = 0.55 * self.velocity + 0.45 * measured_velocity
        self.center = new_center
        self.box = (x, y, w, h)
        self.missed = 0
        return self.box, tuple(map(int, np.round(self.center))), True

    def _miss(self, prediction):
        self.missed += 1
        if self.center is not None and self.missed <= self.max_missed:
            self.center = prediction
            self.box = None
            return None, tuple(map(int, np.round(self.center))), False
        if self.missed > self.max_missed:
            self.center = None
            self.velocity[:] = 0
            self.box = None
        return None, None, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="input video")
    parser.add_argument("--output", type=Path, help="optional path for annotated output video")
    parser.add_argument("--no-display", action="store_true", help="process without opening preview windows")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise SystemExit(f"Could not create output video: {args.output}")

    tracker = BallTracker()
    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            mask, candidates = find_ball_candidates(frame)
            box, center, detected = tracker.update(candidates)

            view = frame.copy()
            # Candidate boxes are thin blue; the selected detection is green.
            for x, y, w, h, _ in candidates:
                cv2.rectangle(view, (x, y), (x + w, y + h), (255, 100, 0), 1)
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if center is not None:
                cv2.circle(view, center, 3, (0, 255, 255), -1)
                label = "ball" if detected else "predicted"
                cv2.putText(view, label, (center[0] + 5, center[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                            cv2.LINE_AA)
            cv2.putText(view, f"frame {frame_number}", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)

            if writer is not None:
                writer.write(view)
            if not args.no_display:
                cv2.imshow("Traditional ball detector", view)
                cv2.imshow("Bright-region mask", mask)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if args.output:
        print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
