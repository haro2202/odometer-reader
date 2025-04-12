import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
from cluster_utils import *
from PIL import Image, ImageDraw
from pathlib import Path
import tempfile

# --- Configuration ---
st.set_page_config(page_title="Odometer Reader", layout="wide")

st.sidebar.header("🔧 DBSCAN Parameters")
DBSCAN_A = {
    "eps": st.sidebar.slider("DBSCAN A - eps", 5, 50, 35),
    "min_samples": st.sidebar.slider("DBSCAN A - min_samples", 1, 10, 2)
}
DBSCAN_B = {
    "eps": st.sidebar.slider("DBSCAN B - eps", 5, 50, 20),
    "min_samples": st.sidebar.slider("DBSCAN B - min_samples", 1, 10, 2)
}

# --- Load models ---
od_model, digit_model = load_models()
mileage_labels = load_groundtruth()


def draw_odometer_box(image_path, odometer_box, outline="red", width=3):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    if odometer_box:
        draw.rectangle(odometer_box, outline=outline, width=width)
    return img


def draw_cluster_boxes(image, clusters, color="blue"):
    draw = ImageDraw.Draw(image)
    for cluster in clusters:
        boxes = cluster.get("bbox", [])
        if boxes:
            draw.rectangle(boxes, outline=color, width=3)
    return image


def process_image_pipeline(img_path: Path, title: str):
    st.image(str(img_path), caption=f"{title} - Step 1: Uploaded Image", use_column_width=True)

    od_box = detect_odometer_boxes_batch(od_model, img_path)
    if not od_box:
        st.error("❌ No odometer detected.")
        return

    od_img = draw_odometer_box(img_path, od_box)
    st.image(od_img, caption="Step 2: Odometer Detection", use_column_width=True)

    crop_path = crop_odometer_regions(img_path, od_box)
    st.image(crop_path, caption="Step 3: Cropped Region", use_column_width=True)

    digits = detect_digits_batch(digit_model, crop_path)
    angle = estimate_rotation_mode_angle(digits)

    crop_predicted = draw_digit_boxes(Image.open(crop_path).copy(), digits)
    st.image(crop_predicted, caption=f"Step 3: Predicted Digits", use_column_width=True)

    rotated_image_path = rotate_odometer_regions(crop_path, angle)

    rotated_image_digits = detect_digits_batch(digit_model, rotated_image_path, squeeze_y=False)

    rotated_digits = rotate_digits(digits, crop_path, angle, True)

    rotated_digits = merge_digit_predictions(rotated_digits, rotated_image_digits)

    rotated_img = draw_digit_boxes(Image.open(rotated_image_path).copy(), rotated_digits)
    st.image(rotated_img, caption=f"Step 4: Rotated Digits (angle={angle:.1f}°)", use_column_width=True)

    clusters1 = cluster_digits(rotated_digits, **DBSCAN_A)
    clusters2 = cluster_digits(rotated_digits, **DBSCAN_B)

    cluster_img1 = draw_cluster_boxes(Image.open(rotated_image_path).copy(), clusters1)
    cluster_img2 = draw_cluster_boxes(Image.open(rotated_image_path).copy(), clusters2)

    st.image(cluster_img1, caption="Step 5: Clusters (Top-1)", use_column_width=True)
    st.image(cluster_img2, caption="Step 6: Clusters (Top-2)", use_column_width=True)

    top1 = norm(clusters1[0]["digits"]) if clusters1 else None
    top2 = norm(clusters2[0]["digits"]) if clusters2 else None
    st.success(f"Top-1 Prediction: `{top1}`")
    if top2:
        st.info(f"Top-2 Candidate: `{top2}`")

    truth = norm(mileage_labels.get(img_path.name))
    if truth:
        st.write(f"Ground Truth: `{truth}`")
        if truth in [top1, top2]:
            st.success("✅ Top-2 Match Correct!")
        else:
            st.warning("⚠️ Ground truth not in top-2.")


# --- User Image Upload ---
st.title("📷 Odometer Reader with Visual Steps")
user_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if user_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(user_file.read())
        img_path = Path(tmp.name)
    process_image_pipeline(img_path, "User Uploaded")


# --- Batch Evaluation ---
st.header("📊 Batch Evaluation on Test Set")
RESULTS_PATH = Path("datasets/trodo/test_set/results.csv")
if st.button("Update Results") or not RESULTS_PATH.exists():
    with st.spinner("Evaluating test set..."):
        test_dir = Path("datasets/trodo/test_set/images")
        image_paths = sorted(test_dir.glob("*.jpg"))
        results = []
        for sel_path in image_paths:
            box = detect_odometer_boxes_batch(od_model, sel_path)
            crop_path = crop_odometer_regions(sel_path, box)
            digits = detect_digits_batch(digit_model, crop_path)
            angle = estimate_rotation_mode_angle(digits)
            rotated_path = rotate_odometer_regions(crop_path, angle)
            rotated_image_digits = detect_digits_batch(digit_model, rotated_path)
            rotated_digits = rotate_digits(digits, crop_path, angle, True)

            rotated_digits = merge_digit_predictions(rotated_digits, rotated_image_digits)

            clusters1 = cluster_digits(rotated_digits, **DBSCAN_A)
            clusters2 = cluster_digits(rotated_digits, **DBSCAN_B)

            top1 = norm(clusters1[0]["digits"]) if clusters1 else None
            top2 = norm(clusters2[0]["digits"]) if clusters2 else None
            truth = norm(mileage_labels.get(sel_path.name))

            results.append({
                "filename": sel_path.name,
                "top1": top1,
                "top2": top2,
                "truth": truth,
                "Top-1 Match": top1 == truth,
                "Top-2 Match": truth in [top1, top2]
            })
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

if RESULTS_PATH.exists():
    df = pd.read_csv(RESULTS_PATH)
    st.metric("Top-1 Accuracy", f"{df['Top-1 Match'].mean() * 100:.2f}%")
    st.metric("Top-2 Accuracy", f"{df['Top-2 Match'].mean() * 100:.2f}%")

    st.subheader("Results Table")
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection("single", use_checkbox=True)
    grid_response = AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=400,
        theme="alpine"
    )
    selected = grid_response["selected_rows"]
    if selected is not None:
        st.session_state.selected_file = selected.to_dict('records')[0]["filename"]

# --- Visual for Selected File ---
if "selected_file" in st.session_state:
    st.subheader(f"📂 Selected File: {st.session_state.selected_file}")
    sel_path = Path("datasets/trodo/test_set/images") / st.session_state.selected_file
    process_image_pipeline(sel_path, "Test Set")
