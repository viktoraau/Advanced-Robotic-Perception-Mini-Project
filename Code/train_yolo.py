
from pathlib import Path
from ultralytics import YOLO
import torch


def main():

    # ==================================================
    # PATHS
    # ==================================================

    PROJECT_DIR = Path(__file__).resolve().parent

    DATA_YAML = PROJECT_DIR / "YOLO_Dataset" / "data.yaml"

    RUNS_DIR = PROJECT_DIR / "runs"

    # ==================================================
    # VERIFY GPU
    # ==================================================

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    print("Using GPU:", torch.cuda.get_device_name(0))

    print(
        "Available GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3, 2
        ),
        "GB"
    )

    # ==================================================
    # LOAD MODEL
    # ==================================================

    model = YOLO("yolov8n.pt")

    print("\nStarting YOLO training...")

    # ==================================================
    # TRAINING
    # ==================================================

    results = model.train(

        # Dataset
        data=str(DATA_YAML),

        # Training parameters
        epochs=75,
        imgsz=1280,
        batch=2,

        # Hardware
        device=0,
        workers=0,

        # Training configuration
        pretrained=True,
        optimizer="auto",
        patience=15,

        # Memory management
        cache=False,
        amp=True,

        # Reproducibility
        seed=42,

        # Output
        project=str(RUNS_DIR),
        name="soccer_ball_yolov8n",
        exist_ok=True,

        # Additional settings
        plots=True,
        close_mosaic=10
    )

    print("\nTRAINING COMPLETE!")

    print("Results saved in:")
    print(RUNS_DIR / "soccer_ball_yolov8n")


# ==================================================
# WINDOWS MAIN GUARD
# ==================================================

if __name__ == "__main__":
    main()
