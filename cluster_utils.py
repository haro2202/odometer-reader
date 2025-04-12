from pathlib import Path
import json
import tempfile
import os
import math
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import logging
import cv2

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)


def load_models():
    return (
        YOLO("runs/train/odometer_detector/weights/best.pt"),
        YOLO("runs/train/digit_detector/weights/best.pt")
    )


def load_groundtruth():
    path = Path("datasets/trodo/mileage_labels.json")
    return json.load(open(path)) if path.exists() else {}


def norm(val):
    try:
        if "." in str(val):
            parts = str(val).split(".")
            if len(parts[1]) == 1:
                return parts[0] + parts[1]  # e.g., 141.1 → 1411
        return str(int(float(val)))
    except:
        return None


def detect_odometer_boxes_batch(model, image_paths):
    if isinstance(image_paths, (str, Path)):
        results = model.predict(str(image_paths))
        for b in results[0].boxes.data:
            if int(b[5]) == 0:
                x1, y1, x2, y2 = map(int, b[:4])
                return x1, y1, x2, y2
        return None

    od_boxes = {}
    results = model.predict([str(p) for p in image_paths])
    for idx, (img_path, result) in enumerate(zip(image_paths, results)):
        box = None
        for b in results[idx].boxes.data:
            if int(b[5]) == 0:
                x1, y1, x2, y2 = map(int, b[:4])
                box = (x1, y1, x2, y2)
                break
        od_boxes[str(img_path)] = box
    return od_boxes


def crop_odometer_regions(image_paths, od_boxes):
    if isinstance(image_paths, (str, Path)):
        img = Image.open(image_paths).convert("RGB")
        cropped = img.crop(od_boxes)
        tmp = os.path.join(tempfile.gettempdir(), f"crop_{Path(image_paths).name}")
        cropped.save(tmp)
        return tmp

    cropped_paths = {}
    for path in image_paths:
        box = od_boxes.get(str(path))
        if not box:
            continue
        img = Image.open(path).convert("RGB")
        cropped = img.crop(box)
        tmp = os.path.join(tempfile.gettempdir(), f"crop_{Path(path).name}")
        cropped.save(tmp)
        cropped_paths[str(path)] = tmp
    return cropped_paths


def rotate_odometer_regions(image_paths, angles, prefix="rotated_"):
    if isinstance(image_paths, (str, Path)):
        img = Image.open(image_paths).convert("RGB")
        rotated = img.rotate(angles, expand=True)
        tmp = os.path.join(tempfile.gettempdir(), f"{prefix}{Path(image_paths).name}")
        rotated.save(tmp)
        return tmp

    rotated_paths = {}
    for path in image_paths:
        angle = angles.get(str(path))
        if angle is None:
            continue
        img = Image.open(path).convert("RGB")
        rotated = img.rotate(angle, expand=True)
        tmp = os.path.join(tempfile.gettempdir(), f"{prefix}{Path(path).name}")
        rotated.save(tmp)
        rotated_paths[str(path)] = tmp
    return rotated_paths


def rotate_digits(digits, image_path, angle_deg, expand=False):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    if not expand:
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        new_w, new_h = w, h
    else:
        angle_rad = math.radians(angle_deg)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))

        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)

        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

    def transform_point(x, y):
        point = np.dot(M, np.array([x, y, 1]))
        return float(point[0]), float(point[1])

    rotated_digits = []
    for d in digits:
        # Transform center
        x_new, y_new = transform_point(d["x"], d["y"])

        # Transform all 4 corners of bbox
        x1, y1, x2, y2 = d["bbox"]
        corners = [
            transform_point(x1, y1),  # top-left
            transform_point(x2, y1),  # top-right
            transform_point(x2, y2),  # bottom-right
            transform_point(x1, y2),  # bottom-left
        ]
        xs, ys = zip(*corners)
        new_bbox = [min(xs), min(ys), max(xs), max(ys)]

        d_copy = d.copy()
        d_copy["x"], d_copy["y"] = x_new, y_new
        d_copy["bbox"] = new_bbox
        rotated_digits.append(d_copy)

    return rotated_digits


def detect_digits_batch(model, image_paths, squeeze_y=False):
    if isinstance(image_paths, (str, Path)):
        results = model.predict(str(image_paths), iou=.4, conf=.4)
        digits = []
        for b in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = b.tolist()
            digits.append({
                "class": int(cls),
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "bbox": [x1 + (x2 - x1) / 5, y1 + (y2 - y1) / 5, x2 - (x2 - x1) / 5,
                         y2 - (y2 - y1) / 5] if squeeze_y else [x1, y1, x2, y2],
                "conf": conf
            })
        return digits

    digit_dict = {}
    results = model.predict([str(p) for p in image_paths])
    for idx, (img_path, res) in enumerate(zip(image_paths, results)):
        digits = []
        for b in results[idx].boxes.data:
            x1, y1, x2, y2, conf, cls = b.tolist()
            digits.append({
                "class": int(cls),
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })
        digit_dict[str(img_path)] = digits
    return digit_dict


def estimate_rotation_mode_angle(digits, bin_size=3):
    if len(digits) < 2:
        return 0.0
    centers = np.array([[d["x"], d["y"]] for d in digits])
    angles = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            if dx == 0 and dy == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            angle = (angle + 90) % 180 - 90
            angles.append(round(angle / bin_size) * bin_size)
    if not angles:
        return 0.0
    most_common_angle, _ = Counter(angles).most_common(1)[0]
    return most_common_angle


def cluster_digits(
        digits,
        size_factor_range=(0.7, 1.5),
        bin_size=2.0,
        y_tolerance_ratio=0.8,
        max_x_gap_ratio=2.0,
        **kwargs
):
    if not digits:
        return []

    # --- Common diagonal (size) ---
    diagonals = [
        math.hypot(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1])
        for d in digits
    ]
    hist, bin_edges = np.histogram(
        diagonals, bins=np.arange(0, max(diagonals) + bin_size, bin_size)
    )
    max_bin_index = np.argmax(hist)
    common_diag = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    coords = np.array([[d["x"], d["y"]] for d in digits])
    clustering = DBSCAN(**kwargs).fit(coords)
    labels = clustering.labels_

    best_group = None

    for label in np.unique(labels):
        members = [d for i, d in enumerate(digits) if labels[i] == label]

        # --- Filter by size ---
        filtered = []
        for d in members:
            w = d["bbox"][2] - d["bbox"][0]
            h = d["bbox"][3] - d["bbox"][1]
            diag = math.hypot(w, h)
            if size_factor_range[0] * common_diag <= diag <= size_factor_range[1] * common_diag:
                filtered.append(d)
        if not filtered:
            continue

        # --- Filter by vertical alignment ---
        y_centers = [(d["bbox"][1] + d["bbox"][3]) / 2 for d in filtered]
        median_y = np.median(y_centers)
        avg_height = np.mean([d["bbox"][3] - d["bbox"][1] for d in filtered])
        y_tol = y_tolerance_ratio * avg_height
        filtered = [
            d for d, y in zip(filtered, y_centers)
            if abs(y - median_y) <= y_tol
        ]
        if not filtered:
            continue

        # --- Sort by X and split if too far ---
        filtered = sorted(filtered, key=lambda d: d["x"])
        x_centers = [(d["bbox"][0] + d["bbox"][2]) / 2 for d in filtered]
        gaps = [x2 - x1 for x1, x2 in zip(x_centers, x_centers[1:])]
        avg_gap = np.median(gaps) if gaps else 0

        groups = []
        current_group = [filtered[0]]
        for i in range(1, len(filtered)):
            prev_x = (filtered[i - 1]["bbox"][0] + filtered[i - 1]["bbox"][2]) / 2
            curr_x = (filtered[i]["bbox"][0] + filtered[i]["bbox"][2]) / 2
            if abs(curr_x - prev_x) <= max_x_gap_ratio * avg_gap:
                current_group.append(filtered[i])
            else:
                groups.append(current_group)
                current_group = [filtered[i]]
        groups.append(current_group)

        # --- Find the best group (most digits) ---
        for g in groups:
            if not g:
                continue
            digits_str = ''.join(str(d["class"]) for d in g)
            avg_conf = np.mean([d.get("conf", 0.9) for d in g])
            x1 = min(d["bbox"][0] for d in g)
            y1 = min(d["bbox"][1] for d in g)
            x2 = max(d["bbox"][2] for d in g)
            y2 = max(d["bbox"][3] for d in g)
            group_info = {
                "digits": digits_str,
                "bbox": [x1, y1, x2, y2],
                "area": (x2 - x1) * (y2 - y1),
                "avg_conf": avg_conf,
            }
            if (
                    best_group is None
                    or len(group_info["digits"]) > len(best_group["digits"])
                    or (len(group_info["digits"]) == len(best_group["digits"]) and group_info["avg_conf"] >
                        best_group["avg_conf"])
            ):
                best_group = group_info

    return [best_group] if best_group else []


def draw_digit_boxes(image: Image.Image, digits: list, color="lime", font_size=14) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    for d in digits:
        box = d.get("bbox")
        if box:
            draw.rectangle(box, outline=color, width=2)
    for d in digits:
        box = d.get("bbox")
        cls = d.get("class", None)
        if box:
            if cls is not None:
                draw.text((box[0], box[1]), str(cls), fill="blue", font=font)
    return image


def bbox_iou(box1, box2):
    """Calculate IOU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def merge_digit_predictions(preds_a, preds_b, iou_threshold=0.3):
    merged = []
    used_b = set()

    for a in preds_a:
        best_match = None
        best_iou = 0
        for i, b in enumerate(preds_b):
            if i in used_b:
                continue
            iou = bbox_iou(a["bbox"], b["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = (i, b)

        if best_match and best_iou >= iou_threshold:
            i, b = best_match
            used_b.add(i)
            merged.append({
                "class": a["class"],
                "conf": a["conf"],
                "x": b["x"],
                "y": b["y"],
                "bbox": b["bbox"]
            })
        else:
            # Fallback: use everything from A if no match found
            merged.append({
                "class": a["class"],
                "conf": a["conf"],
                "x": a["x"],
                "y": a["y"],
                "bbox": a["bbox"]
            })

    return merged
