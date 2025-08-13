import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
from cluster_utils import *
from PIL import Image, ImageDraw
from pathlib import Path
import tempfile

# --- Configuration ---
st.set_page_config(page_title="Odometer Reader", layout="wide")

st.sidebar.header("🔧 Detection Parameters")
MIN_CONFIDENCE = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.4, 0.05)
DBSCAN_PARAMS = {
    "eps": st.sidebar.slider("DBSCAN - eps", 5, 50, 30),
    "min_samples": st.sidebar.slider("DBSCAN - min_samples", 1, 10, 2)
}

# --- Load models ---
od_model1, od_model2, digit_model = load_models()
mileage_labels = load_groundtruth()


def filter_digits_by_confidence(digits, min_confidence=0.5):
    """Lọc các chữ số theo confidence threshold và loại bỏ duplicates"""
    if not digits:
        return []
    
    # Bước 1: Lọc theo confidence
    filtered = [d for d in digits if d.get("conf", 0) >= min_confidence]
    
    # Bước 2: Loại bỏ duplicates tại cùng vị trí (nếu có overlap lớn)
    final_digits = []
    for i, digit in enumerate(filtered):
        is_duplicate = False
        for j, existing in enumerate(final_digits):
            # Tính khoảng cách giữa 2 center points
            dist = ((digit["x"] - existing["x"]) ** 2 + (digit["y"] - existing["y"]) ** 2) ** 0.5
            # Nếu quá gần và confidence thấp hơn thì bỏ qua
            if dist < 5:  # threshold khoảng cách
                if digit["conf"] <= existing["conf"]:
                    is_duplicate = True
                    break
                else:
                    # Thay thế digit có confidence cao hơn
                    final_digits[j] = digit
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            final_digits.append(digit)
    
    return final_digits


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
    st.image(str(img_path), caption=f"{title} - Step 1: Uploaded Image", use_container_width=True)

    od_box = detect_odometer_boxes_batch(od_model1, img_path)
    if not od_box:
        od_box = detect_odometer_boxes_batch(od_model2, img_path)
        if not od_box:
            st.error("❌ No odometer detected.")
            return
    

    od_img = draw_odometer_box(img_path, od_box)
    st.image(od_img, caption="Step 2: Odometer Detection", use_container_width=True)

    crop_path = crop_odometer_regions(img_path, od_box)
    st.image(crop_path, caption="Step 3: Cropped Region", use_container_width=True)

    digits = detect_digits_batch(digit_model, crop_path)
    print(f"Raw detected digits: {len(digits)} detections")
    
    # Áp dụng confidence filtering
    digits = filter_digits_by_confidence(digits, MIN_CONFIDENCE)
    print(f"Filtered digits (conf >= {MIN_CONFIDENCE}): {digits}")
    
    angle = estimate_rotation_mode_angle(digits)

    crop_predicted = draw_digit_boxes(Image.open(crop_path).copy(), digits)
    st.image(crop_predicted, caption=f"Step 4: Filtered Digits (conf >= {MIN_CONFIDENCE:.1f})", use_container_width=True)

    # rotated_image_path = rotate_odometer_regions(crop_path, angle)

    # rotated_image_digits = detect_digits_batch(digit_model, rotated_image_path, squeeze_y=False)

    # rotated_digits = rotate_digits(digits, crop_path, angle, True)

    # rotated_digits = merge_digit_predictions(rotated_digits, rotated_image_digits)

    # rotated_img = draw_digit_boxes(Image.open(rotated_image_path).copy(), rotated_digits)
    # st.image(rotated_img, caption=f"Step 4: Rotated Digits (angle={angle:.1f}°)", use_container_width=True)

    clusters = cluster_digits(digits, **DBSCAN_PARAMS)

    cluster_img = draw_cluster_boxes(Image.open(crop_path).copy(), clusters)
    st.image(cluster_img, caption="Step 5: Clustered Digits", use_container_width=True)

    prediction = norm(clusters[0]["digits"]) if clusters else None
    st.success(f"Prediction: `{prediction}`")

    truth = norm(mileage_labels.get(img_path.name))
    if truth:
        st.write(f"Ground Truth: `{truth}`")
        if truth == prediction:
            st.success("✅ Prediction Correct!")
        else:
            st.warning("⚠️ Prediction does not match ground truth.")


# --- User Image Upload ---
st.title("📷 Odometer Reader with Visual Steps")
user_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if user_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(user_file.read())
        img_path = Path(tmp.name)
    process_image_pipeline(img_path, "User Uploaded")


# # --- Batch Evaluation ---
# st.header("📊 Batch Evaluation on Test Set")
# RESULTS_PATH = Path("datasets/trodo/test_set/results.csv")
# if st.button("Update Results") or not RESULTS_PATH.exists():
#     with st.spinner("Evaluating test set..."):
#         test_dir = Path("datasets/trodo/test_set/images")
#         image_paths = sorted(test_dir.glob("*.jpg"))
#         results = []
#         for sel_path in image_paths:
#             box = detect_odometer_boxes_batch(od_model, sel_path)
#             crop_path = crop_odometer_regions(sel_path, box)
#             digits = detect_digits_batch(digit_model, crop_path)
            
#             # Áp dụng confidence filtering
#             digits = filter_digits_by_confidence(digits, MIN_CONFIDENCE)
            
#             angle = estimate_rotation_mode_angle(digits)
#             rotated_path = rotate_odometer_regions(crop_path, angle)
#             rotated_image_digits = detect_digits_batch(digit_model, rotated_path)
            
#             # Lọc cả rotated digits
#             rotated_image_digits = filter_digits_by_confidence(rotated_image_digits, MIN_CONFIDENCE)
            
#             rotated_digits = rotate_digits(digits, crop_path, angle, True)

#             rotated_digits = merge_digit_predictions(rotated_digits, rotated_image_digits)

#             clusters = cluster_digits(rotated_digits, **DBSCAN_PARAMS)

#             prediction = norm(clusters[0]["digits"]) if clusters else None
#             truth = norm(mileage_labels.get(sel_path.name))

#             results.append({
#                 "filename": sel_path.name,
#                 "prediction": prediction,
#                 "truth": truth,
#                 "Match": prediction == truth
#             })
#         pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

# if RESULTS_PATH.exists():
#     df = pd.read_csv(RESULTS_PATH)
#     st.metric("Top-1 Accuracy", f"{df['Top-1 Match'].mean() * 100:.2f}%")
#     st.metric("Top-2 Accuracy", f"{df['Top-2 Match'].mean() * 100:.2f}%")

#     st.subheader("Results Table")
#     gb = GridOptionsBuilder.from_dataframe(df)
#     gb.configure_selection("single", use_checkbox=True)
#     grid_response = AgGrid(
#         df,
#         gridOptions=gb.build(),
#         update_mode=GridUpdateMode.SELECTION_CHANGED,
#         height=400,
#         theme="alpine"
#     )
#     selected = grid_response["selected_rows"]
#     if selected is not None:
#         st.session_state.selected_file = selected.to_dict('records')[0]["filename"]

# # --- Visual for Selected File ---
# if "selected_file" in st.session_state:
#     st.subheader(f"📂 Selected File: {st.session_state.selected_file}")
#     sel_path = Path("datasets/trodo/test_set/images") / st.session_state.selected_file
#     process_image_pipeline(sel_path, "Test Set")
