# Leaf Classification with an "Unknown" Class

A beginner-friendly computer vision project that classifies leaf photos into **5 known classes plus an "unknown" class**, using transfer learning (MobileNetV2) and a Random Forest.

---

## 1. Project Overview

- **Goal:** classify a leaf photo into one of five known classes, or say "unknown" if it does not match any of them.
- **Approach:** pretrained MobileNetV2 extracts image features, and a Random Forest classifies them.
- **Unknown class:** handled in two ways: (1) training examples from other leaf folders, and (2) a confidence threshold at prediction time.

## 2. Dataset

- **Source:** [Plant Leaves for Image Classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification) (Kaggle, `csafrit2`)
- **Download:** done automatically in the code with `kagglehub`. The full dataset is not stored in this repo because of its size.
- **Known classes:** the five largest folders: `<FILL IN: list your 5 classes from outputs/results.json>`
- **Unknown class:** images sampled from all the other folders.
- **Size used:** up to 100 images per class.
- **Metadata:** `outputs/metadata.csv` lists every image path, original folder name, and final label.

## 3. Repository Structure

```
.
├── leaf_classification.ipynb    # main notebook (same code as the .py file)
├── leaf_classification.py       # same code as a script
├── requirements.txt
├── README.md
├── sample_images/               # 3 example photos per class
└── outputs/
    ├── metadata.csv
    ├── results.json             # all metrics
    ├── class_counts.png
    ├── image_sizes.png
    ├── color_distribution.png
    ├── montage.png
    ├── confusion_matrix.png
    └── confusion_matrix_threshold.png
```

## 4. Exploratory Data Analysis (EDA)

| Plot | What it shows |
|---|---|
| `class_counts.png` | Number of images in each class |
| `image_sizes.png` | Width and height of the photos |
| `color_distribution.png` | Mean red, green and blue values per class |
| `montage.png` | Sample photos from each class |

**Observations:** `<FILL IN: 2-3 sentences, e.g. are the classes balanced, are image sizes consistent, do some classes look very similar in color?>`

## 5. Method

1. **Metadata:** build a CSV of images and labels.
2. **Preprocessing:** resize to 224×224, scale pixels to [-1, 1] (MobileNetV2 normalization).
3. **Split:** 80% train and 20% test, stratified by class.
4. **Features:** MobileNetV2 (ImageNet weights, no top layer, average pooling) gives a 1280-value vector per image.
5. **Classifier:** Random Forest with 300 trees trained on those features.
6. **Unknown handling:**
   - an explicit "unknown" class built from other leaf folders
   - a confidence threshold of 0.5: if the highest class probability is below it, predict "unknown"

## 6. Results

Copy the numbers from `outputs/results.json`.

| Metric | Without threshold | With threshold (0.5) |
|---|---|---|
| Accuracy | `<FILL IN>` | `<FILL IN>` |
| Macro-F1 | `<FILL IN>` | `<FILL IN>` |

**Per-class recall**

| Class | Recall |
|---|---|
| `<class 1>` | `<FILL IN>` |
| `<class 2>` | `<FILL IN>` |
| `<class 3>` | `<FILL IN>` |
| `<class 4>` | `<FILL IN>` |
| `<class 5>` | `<FILL IN>` |
| unknown | `<FILL IN>` |

![Confusion matrix](outputs/confusion_matrix.png)

## 7. Discussion: What Gets Confused and Why

`<FILL IN after running the code. Use the "Most confused pairs" printed in the notebook and the confusion matrix. Cover:>`

- Which classes are confused most often (for example: `Class A` predicted as `Class B`).
- Likely reasons: similar leaf shape, similar color or texture, similar background, or few training images.
- How the "unknown" class behaves: does it absorb mistakes, or do real leaves get wrongly called unknown?
- What the confidence threshold changed (more unknowns caught, or more correct leaves rejected).

## 8. Limitations

- Small dataset (up to 100 images per class), so results can change with a different random split.
- The "unknown" class only contains leaves from this dataset. Completely different objects (for example a hand or a table) are not covered.
- Photos in this dataset may have clean backgrounds, so real-world phone photos may perform worse.
- Frozen features were used, with no fine-tuning of MobileNetV2.

## 9. Possible Extensions

- Fine-tune MobileNetV2 or try ResNet.
- Compare with a beginner model using hand-made color, texture and shape features with SVM.
- Classify healthy vs unhealthy leaves.
- Add photos of local tree species taken with a phone.

## 10. How to Run

**Google Colab (easiest)**
1. Upload `leaf_classification.ipynb` to Colab.
2. Uncomment the `!pip install` line in the first cell and run it.
3. Run all cells. Colab can download the Kaggle dataset through `kagglehub`.

**Locally**
```bash
pip install -r requirements.txt
python leaf_classification.py
```
Outputs are saved to `outputs/` and `sample_images/`.

**Predict your own photo:** use the `predict_photo("my_leaf.jpg")` function at the end of the notebook. It returns the predicted class and confidence.

## 11. Tools Used

Python, TensorFlow/Keras (MobileNetV2), scikit-learn (Random Forest and metrics), pandas, matplotlib, Pillow, kagglehub.

## 12. Credits

- Dataset: Kaggle, [csafrit2/plant-leaves-for-image-classification](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification)
- Pretrained model: MobileNetV2 (ImageNet weights, via Keras)
