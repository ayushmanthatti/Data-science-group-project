# %% 1. Install (run once in Colab/Jupyter, then comment out)
# !pip install -q kagglehub tensorflow scikit-learn pandas matplotlib pillow joblib

# %% 2. Imports and settings
import os, shutil, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
from PIL import Image

SEED = 42
N_KNOWN = 5            # number of known classes (the 6th class is "unknown")
MAX_PER_CLASS = 100    # keep the project small and fast
THRESHOLD = 0.5        # below this confidence, predict "unknown"
IMG_SIZE = 224         # MobileNetV2 input size

random.seed(SEED)
np.random.seed(SEED)
os.makedirs("outputs", exist_ok=True)

# %% 3. Download dataset from Kaggle
path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")
print("Path to dataset files:", path)

# %% 4. Build metadata CSV (class = name of the image's parent folder)
rows = []
for root, _, files in os.walk(path):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append({"path": os.path.join(root, f),
                         "raw_class": os.path.basename(root)})
df = pd.DataFrame(rows)
print("Total images:", len(df), "| Folders found:", df["raw_class"].nunique())
print(df["raw_class"].value_counts())

# Known classes = the N biggest folders. Everything else becomes "unknown".
# TIP: to choose your own five, replace this line with a list of folder names,
# e.g. known = ["Mango", "Neem", "Basil", "Guava", "Jamun"]
known = df["raw_class"].value_counts().head(N_KNOWN).index.tolist()
print("\nKnown classes:", known)

df["label"] = df["raw_class"].where(df["raw_class"].isin(known), "unknown")

# Cap each class at MAX_PER_CLASS images ("unknown" is sampled from many folders)
df = pd.concat([g.sample(min(len(g), MAX_PER_CLASS), random_state=SEED)
                for _, g in df.groupby("label")]).reset_index(drop=True)

df.to_csv("outputs/metadata.csv", index=False)
print("\nImages per class used in the project:")
print(df["label"].value_counts())

# %% 5. EDA: class counts, image sizes, color distribution, sample montage
# 5a. Class counts
df["label"].value_counts().plot(kind="bar", title="Images per class")
plt.ylabel("Number of images")
plt.tight_layout(); plt.savefig("outputs/class_counts.png"); plt.show()

# 5b. Image sizes
sizes = np.array([Image.open(p).size for p in df["path"]])
print("Width  min/mean/max:", sizes[:, 0].min(), int(sizes[:, 0].mean()), sizes[:, 0].max())
print("Height min/mean/max:", sizes[:, 1].min(), int(sizes[:, 1].mean()), sizes[:, 1].max())
plt.scatter(sizes[:, 0], sizes[:, 1], alpha=0.4)
plt.xlabel("Width (px)"); plt.ylabel("Height (px)"); plt.title("Image sizes")
plt.tight_layout(); plt.savefig("outputs/image_sizes.png"); plt.show()

# 5c. Color distribution (mean R, G, B per class)
def mean_rgb(p):
    img = Image.open(p).convert("RGB").resize((64, 64))
    return np.array(img).mean(axis=(0, 1))

df[["R", "G", "B"]] = np.stack(df["path"].apply(mean_rgb))
color_table = df.groupby("label")[["R", "G", "B"]].mean().round(1)
print(color_table)
color_table.plot(kind="bar", color=["red", "green", "blue"],
                 title="Mean color per class")
plt.tight_layout(); plt.savefig("outputs/color_distribution.png"); plt.show()

# 5d. Sample montage (one photo per class, 2 rows)
labels = sorted(df["label"].unique())
fig, axes = plt.subplots(2, len(labels), figsize=(3 * len(labels), 6))
for col, label in enumerate(labels):
    samples = df[df["label"] == label].sample(2, random_state=SEED)["path"].tolist()
    for row in range(2):
        axes[row, col].imshow(Image.open(samples[row]))
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(label[:20], fontsize=9)
plt.tight_layout(); plt.savefig("outputs/montage.png"); plt.show()

# %% 6. Preprocessing pipeline: resize -> normalize -> train/test split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

def preprocess(img_path):
    """Open image, resize to 224x224, scale pixels to [-1, 1] (MobileNetV2 style)."""
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return preprocess_input(np.array(img, dtype=np.float32))

y = df["label"].values
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
print("Train images:", len(idx_train), "| Test images:", len(idx_test))

# %% 7. Transfer learning: MobileNetV2 (pretrained on ImageNet) as feature extractor
base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))

def extract_features(paths, batch_size=32):
    """Process images in batches to keep memory use low."""
    out = []
    for i in range(0, len(paths), batch_size):
        batch = np.stack([preprocess(p) for p in paths[i:i + batch_size]])
        out.append(base.predict(batch, verbose=0))
    return np.vstack(out)

paths = df["path"].tolist()
X_train = extract_features([paths[i] for i in idx_train])
X_test = extract_features([paths[i] for i in idx_test])
y_train, y_test = y[idx_train], y[idx_test]
print("Feature shape (train):", X_train.shape)

# %% 8. Train Random Forest on the extracted features
from sklearn.ensemble import RandomForestClassifier
import joblib

clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
clf.fit(X_train, y_train)
joblib.dump(clf, "outputs/random_forest.joblib")

# %% 9. Evaluation: accuracy, macro-F1, per-class recall, confusion matrix
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
recalls = recall_score(y_test, pred, average=None, labels=clf.classes_)

print("Accuracy:", round(acc, 3))
print("Macro-F1:", round(macro_f1, 3))
print("\nPer-class recall:")
for c, r in zip(clf.classes_, recalls):
    print(f"  {c}: {r:.2f}")
print("\n", classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred, labels=clf.classes_)
ConfusionMatrixDisplay(cm, display_labels=[c[:15] for c in clf.classes_]).plot(
    xticks_rotation=45, cmap="Greens")
plt.title("Confusion matrix (Random Forest)")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png"); plt.show()

# %% 10. Unknown handling: confidence threshold
# If the model is not confident enough, call the photo "unknown".
proba = clf.predict_proba(X_test)
pred_thr = np.where(proba.max(axis=1) < THRESHOLD, "unknown",
                    clf.classes_[proba.argmax(axis=1)])
acc_thr = accuracy_score(y_test, pred_thr)
f1_thr = f1_score(y_test, pred_thr, average="macro")
print(f"With threshold {THRESHOLD}: Accuracy = {acc_thr:.3f} | Macro-F1 = {f1_thr:.3f}")

cm_thr = confusion_matrix(y_test, pred_thr, labels=clf.classes_)
ConfusionMatrixDisplay(cm_thr, display_labels=[c[:15] for c in clf.classes_]).plot(
    xticks_rotation=45, cmap="Greens")
plt.title(f"Confusion matrix (threshold = {THRESHOLD})")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix_threshold.png"); plt.show()

# %% 11. Which classes get confused? (use this for the README discussion)
confused = []
for i, true_c in enumerate(clf.classes_):
    for j, pred_c in enumerate(clf.classes_):
        if i != j and cm[i, j] > 0:
            confused.append((int(cm[i, j]), true_c, pred_c))
confused.sort(reverse=True)

print("Most confused pairs (true -> predicted):")
for n, t, p in confused[:10]:
    print(f"  {t} -> {p}: {n} times")

# Save all results so you can copy them into the README
results = {
    "known_classes": known,
    "train_images": int(len(idx_train)),
    "test_images": int(len(idx_test)),
    "accuracy": round(float(acc), 3),
    "macro_f1": round(float(macro_f1), 3),
    "per_class_recall": {c: round(float(r), 2) for c, r in zip(clf.classes_, recalls)},
    "threshold": THRESHOLD,
    "accuracy_with_threshold": round(float(acc_thr), 3),
    "macro_f1_with_threshold": round(float(f1_thr), 3),
    "most_confused": [f"{t} -> {p}: {n}" for n, t, p in confused[:10]],
}
with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved outputs/results.json")

# %% 12. Copy sample images into the project (3 per class) for GitHub
for label, g in df.groupby("label"):
    folder = os.path.join("sample_images", label)
    os.makedirs(folder, exist_ok=True)
    for p in g["path"].head(3):
        shutil.copy(p, folder)
print("Sample images saved in sample_images/")

# %% 13. Predict a single new photo
def predict_photo(img_path):
    x = preprocess(img_path)[None]
    feats = base.predict(x, verbose=0)
    p = clf.predict_proba(feats)[0]
    label = "unknown" if p.max() < THRESHOLD else clf.classes_[p.argmax()]
    return label, round(float(p.max()), 2)

# Example (uncomment and use your own photo):
# print(predict_photo("my_leaf.jpg"))
