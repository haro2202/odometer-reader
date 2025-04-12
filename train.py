from ultralytics import YOLO
import json
import multiprocessing


def train_odometer(use_tune=False):
    model = YOLO("yolov8n.pt")  # or 'yolov8s.pt'

    if use_tune:
        print("🔧 Tuning odometer model...")
        results = model.tune(
            data="trodo_stage1.yaml",
            epochs=10,
            imgsz=640,
            batch=16,
            device="0",
            iterations=10
        )
        print("🔁 Tuning complete.\nBest config:\n", results)
    else:
        print("🚀 Training odometer model...")
        results = model.train(
            data="trodo_stage1.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            project="runs/train",
            name="odometer_detector",
            device="0"
        )
        print("✅ Training complete.")
        print(results)


def train_digits(use_tune=False):
    model = YOLO("yolov8n.pt")

    if use_tune:
        print("🔧 Tuning digit model...")
        results = model.tune(
            data="trodo_stage2.yaml",
            epochs=10,
            imgsz=640,
            batch=16,
            device="0",
            iterations=10
        )
        print("🔁 Tuning complete.\nBest config:\n", results)
    else:
        print("🚀 Training digit model...")
        results = model.train(
            data="trodo_stage2.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            project="runs/train",
            name="digit_detector",
            device="0"
        )
        print("✅ Training complete.")
        print(results)


def evaluate(model_path: str, dataset_yaml: str, output_json: str = None):
    model = YOLO(model_path)

    metrics = model.val(data=dataset_yaml)

    print("\n📊 Evaluation Metrics:")
    print(f"Precision      : {metrics.box.mp:.4f}")
    print(f"Recall         : {metrics.box.mr:.4f}")
    print(f"mAP@0.5        : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95   : {[float(m) for m in metrics.box.maps]}")

    if output_json:
        results = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map@0.5": float(metrics.box.map50),
            "map@0.5:0.95": [float(m) for m in metrics.box.maps]
        }
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_json}")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # train_odometer(use_tune=False)

    # train_digits(use_tune=False)

    evaluate(
        model_path="runs/train/odometer_detector/weights/best.pt",
        dataset_yaml="trodo_stage1.yaml",
        output_json="odometer_eval.json"
    )

    evaluate(
        model_path="runs/train/digit_detector/weights/best.pt",
        dataset_yaml="trodo_stage2.yaml",
        output_json="digit_eval.json"
    )
