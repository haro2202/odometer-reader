import json
import xml.etree.ElementTree as ET
from PIL import Image, ImageFile
from pathlib import Path
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

DIGIT_CLASSES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
CLASS_MAP = {**{'odometer': 0}, **{v: int(v) for v in DIGIT_CLASSES}}

RAW_ROOT = Path("datasets/raw/trodo-v01")
OUT_ROOT = Path("datasets/trodo")
ANNOTATIONS = RAW_ROOT / "pascal voc 1.1" / "annotations"
IMAGES = RAW_ROOT / "images"
GROUNDTRUTH_PATH = RAW_ROOT / "ground truth" / "groundtruth.json"

# Output folders
S1_TRAIN_IMG = OUT_ROOT / "stage1_full_odometer/images/train"
S1_VAL_IMG = OUT_ROOT / "stage1_full_odometer/images/val"
S1_TRAIN_LBL = OUT_ROOT / "stage1_full_odometer/labels/train"
S1_VAL_LBL = OUT_ROOT / "stage1_full_odometer/labels/val"

S2_TRAIN_IMG = OUT_ROOT / "stage2_digits_only/images/train"
S2_VAL_IMG = OUT_ROOT / "stage2_digits_only/images/val"
S2_TRAIN_LBL = OUT_ROOT / "stage2_digits_only/labels/train"
S2_VAL_LBL = OUT_ROOT / "stage2_digits_only/labels/val"

TEST_IMG = OUT_ROOT / "test_set/images"
TEST_LBL = OUT_ROOT / "test_set/labels"

for path in [
    S1_TRAIN_IMG, S1_VAL_IMG, S1_TRAIN_LBL, S1_VAL_LBL,
    S2_TRAIN_IMG, S2_VAL_IMG, S2_TRAIN_LBL, S2_VAL_LBL,
    TEST_IMG, TEST_LBL
]:
    path.mkdir(parents=True, exist_ok=True)

# Load mileage values
if GROUNDTRUTH_PATH.exists():
    with open(GROUNDTRUTH_PATH) as f:
        groundtruth = json.load(f)
        groundtruth = {v["image"]: v["mileage"] for v in groundtruth["odometers"]}
else:
    groundtruth = {}

mileage_label_map = {}


def get_mileage(filename):
    return groundtruth.get(filename) or groundtruth.get(filename.replace(".jpg", ""))


def voc_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height


def fix_image_if_needed(path):
    try:
        img = Image.open(path)
        img.verify()
    except Exception:
        try:
            img = Image.open(path).convert("RGB")
            img.save(path)
            print(f"🛠 Fixed: {path.name}")
        except Exception as e:
            print(f"❌ Cannot fix {path.name} — {e}")
            return False
    return True


def convert_and_crop():
    all_xmls = list(ANNOTATIONS.glob("*.xml"))
    random.shuffle(all_xmls)

    N = len(all_xmls)
    N_test = int(N * 0.05)
    N_val = int(N * 0.15)
    N_train = N - N_val - N_test

    train_samples = set(all_xmls[:N_train])
    val_samples = set(all_xmls[N_train:N_train + N_val])
    test_samples = set(all_xmls[N_train + N_val:])

    skipped = []

    for xml_file in all_xmls:
        image_name = xml_file.stem + ".jpg"
        image_path = IMAGES / image_name
        if not image_path.exists():
            print(f"⚠️ Image not found: {image_name}")
            continue

        if not fix_image_if_needed(image_path):
            skipped.append(image_name)
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size
        except Exception as e:
            print(f"❌ Skipping {image_name} — {e}")
            skipped.append(image_name)
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()

        odometer_box = None
        digit_boxes = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            b = obj.find("bndbox")
            bbox = [int(float(b.find(x).text)) for x in ["xmin", "ymin", "xmax", "ymax"]]

            if name == "odometer":
                odometer_box = bbox
            elif name in DIGIT_CLASSES:
                digit_boxes.append((name, bbox))

        if odometer_box is None:
            print(f"⚠️ Skipping {image_name} (no odometer found)")
            skipped.append(image_name)
            continue

        # Select destination folders
        if xml_file in test_samples:
            img_out_path = TEST_IMG / image_name
            lbl_out_path = TEST_LBL / f"{xml_file.stem}.txt"
            s2_img_path = None
            s2_lbl_path = None
        elif xml_file in val_samples:
            img_out_path = S1_VAL_IMG / image_name
            lbl_out_path = S1_VAL_LBL / f"{xml_file.stem}.txt"
            s2_img_path = S2_VAL_IMG / image_name
            s2_lbl_path = S2_VAL_LBL / f"{xml_file.stem}.txt"
        else:
            img_out_path = S1_TRAIN_IMG / image_name
            lbl_out_path = S1_TRAIN_LBL / f"{xml_file.stem}.txt"
            s2_img_path = S2_TRAIN_IMG / image_name
            s2_lbl_path = S2_TRAIN_LBL / f"{xml_file.stem}.txt"

        mileage_value = get_mileage(image_name)
        if mileage_value:
            mileage_label_map[image_name] = mileage_value

        try:
            img.save(img_out_path)
            od_x, od_y, od_w, od_h = voc_to_yolo(odometer_box, img_w, img_h)
            with open(lbl_out_path, "w") as f:
                f.write(f"{CLASS_MAP['odometer']} {od_x:.6f} {od_y:.6f} {od_w:.6f} {od_h:.6f}\n")
        except Exception as e:
            print(f"❌ Stage1 write failed: {image_name} — {e}")
            skipped.append(image_name)
            continue

        # Stage 2 (cropped odometer + digits) only for train/val
        if s2_img_path is not None:
            try:
                cropped = img.crop(odometer_box)
                crop_w, crop_h = cropped.size
                cropped.save(s2_img_path)
            except Exception as e:
                print(f"❌ Cropping failed: {image_name} — {e}")
                skipped.append(image_name)
                continue

            x0, y0 = odometer_box[0], odometer_box[1]
            digit_lines = []
            for name, box in digit_boxes:
                xmin, ymin, xmax, ymax = box
                xmin -= x0
                xmax -= x0
                ymin -= y0
                ymax -= y0
                if xmin < 0 or ymin < 0 or xmax > crop_w or ymax > crop_h:
                    continue
                x, y, w, h = voc_to_yolo([xmin, ymin, xmax, ymax], crop_w, crop_h)
                digit_lines.append(f"{CLASS_MAP[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            if digit_lines:
                with open(s2_lbl_path, "w") as f:
                    f.write("\n".join(digit_lines))

    mileage_out = OUT_ROOT / "mileage_labels.json"
    mileage_out.write_text(json.dumps(mileage_label_map, indent=4))
    print(f"\n✅ Data prepared. Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    print(f"📄 Mileage labels saved to: {mileage_out}")

    if skipped:
        print(f"⚠️ Skipped {len(skipped)} files. See skipped_images.txt")
        with open("skipped_images.txt", "w") as f:
            f.write("\n".join(skipped))


if __name__ == "__main__":
    convert_and_crop()
