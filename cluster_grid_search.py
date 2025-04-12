from tqdm import tqdm
from itertools import product
import pandas as pd
from cluster_utils import *

# --- Define DBSCAN parameter grid ---
eps_values = list(range(20, 40, 1))
min_samples_values = list(range(1, 3))
metrics = ['euclidean']
algorithms = ['auto']
leaf_sizes = [30]
p_values = [None]

param_combinations = list(product(
    eps_values,
    min_samples_values,
    metrics,
    algorithms,
    leaf_sizes,
    p_values
))

image_dir = Path("datasets/trodo/test_set/images")
image_paths = sorted(image_dir.glob("*.jpg"))
truths = load_groundtruth()
od_model, digit_model = load_models()

results = []
for eps, min_samples, metric, algorithm, leaf_size, p in tqdm(param_combinations):
    top1_set = set()
    total = 0

    for sel_path in image_paths:
        box = detect_odometer_boxes_batch(od_model, sel_path)
        crop_path = crop_odometer_regions(sel_path, box)
        digits = detect_digits_batch(digit_model, crop_path)
        if len(digits) < 2:
            continue

        angle = estimate_rotation_mode_angle(digits)
        rotated_image_path = rotate_odometer_regions(crop_path, angle)
        rotated_image_digits = detect_digits_batch(digit_model, rotated_image_path)
        rotated_digits = rotate_digits(digits, crop_path, angle, True)

        rotated_digits = merge_digit_predictions(rotated_digits, rotated_image_digits)

        kwargs = dict(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )

        try:
            clusters = cluster_digits(rotated_digits, **kwargs)
        except Exception:
            continue

        top1 = norm(clusters[0]["digits"]) if clusters else None
        truth = norm(truths.get(sel_path.name))

        total += 1
        if truth == top1:
            top1_set.add(sel_path.name)

    results.append({
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "algorithm": algorithm,
        "leaf_size": leaf_size,
        "p": p,
        "paths": top1_set,
        "Images Processed": total
    })

# Compute Top-1 and Top-2 Accuracy
total_images = len(image_paths)

precomputed = [
    {
        "eps": r["eps"],
        "min_samples": r["min_samples"],
        "metric": r["metric"],
        "algorithm": r["algorithm"],
        "leaf_size": r["leaf_size"],
        "p": r["p"],
        "paths": r["paths"],
        "top1": round(100 * len(r["paths"]) / total_images, 2)
    }
    for r in results
]

new_results = [
    {
        "eps1": r1["eps"], "eps2": r2["eps"],
        "min_samples1": r1["min_samples"], "min_samples2": r2["min_samples"],
        "metric1": r1["metric"], "metric2": r2["metric"],
        "leaf_size1": r1["leaf_size"], "leaf_size2": r2["leaf_size"],
        "p1": r1["p"], "p2": r2["p"],
        "Top-1 Accuracy": r1["top1"],
        "Top-2 Accuracy": round(100 * len(r1["paths"].union(r2["paths"])) / total_images, 2),
        "Images Processed": total_images
    }
    for r1, r2 in product(precomputed, repeat=2)
]


df = pd.DataFrame(new_results)
df.sort_values(by="Top-2 Accuracy", ascending=False, inplace=True)
df.to_csv("grid_search_dbscan.csv", index=False)
print("✅ Full DBSCAN grid search complete. Saved to grid_search_dbscan_full.csv")
