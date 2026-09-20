
# 🌸 Iris Flower Classification

A machine learning project that classifies Iris flowers into one of three species — *Iris-setosa*, *Iris-versicolor*, or *Iris-virginica* — based on sepal and petal measurements. Built as part of a data science internship task.

## 📌 Overview

This project uses the classic **Iris dataset** to build a multi-class classification pipeline. It covers the full workflow: exploratory visualization, preprocessing, model benchmarking, evaluation, and persistence of the final pipeline for reuse.

## 📂 Project Structure

```
Task3/
├── iris_flower_classification.py   # Main script (visualization, training, evaluation, prediction)
├── IRIS.csv                        # Raw dataset (150 flower samples)
├── iris_pipeline.pkl               # Saved model + scaler + label encoder (joblib)
├── iris_data_insights.png          # Pairplot of feature relationships by species
├── model_comparison.png            # Accuracy benchmark across models
├── iris_confusion_matrix.png       # Confusion matrix for the best model
├── requirements.txt                # Python dependencies
└── README.md
```

## 🧠 Approach

1. **Exploratory Visualization**
   - Generates a Seaborn pairplot of all four features (sepal/petal length & width), colored by species, to visualize class separability at a glance.

2. **Preprocessing**
   - Label-encodes the target `species` column (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica` → `0`, `1`, `2`).
   - Uses a **stratified** train/test split to keep equal species representation in both sets.
   - Scales all four numeric features with `StandardScaler`.

3. **Model Training & Evaluation**
   Three classifiers are trained and compared:
   - Logistic Regression
   - Decision Tree
   - Random Forest

   Each is evaluated by validation **accuracy**, and the best-performing model is automatically selected.

4. **Visualization**
   - `iris_data_insights.png` — pairplot of feature distributions and relationships across species.
   - `model_comparison.png` — accuracy comparison across the three models.
   - `iris_confusion_matrix.png` — confusion matrix heatmap for the best model.

5. **Persistence & Inference**
   - The trained model, scaler, and label encoder are saved to `iris_pipeline.pkl` via `joblib`.
   - `predict_species()` demonstrates predicting the species for a single, custom flower measurement.

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

## 🚀 Usage

> ⚠️ **Before running:** the script currently points to a hardcoded local path for the dataset:
> ```python
> FILE_NAME = r"C:\Users\yashw\Yashu\Internship\DATA_SCIENCE\Task3\IRIS.csv"
> ```
> Update `FILE_NAME` to `"IRIS.csv"` (or the correct path on your machine) before running, assuming `IRIS.csv` is in the same folder as the script.

Then run:

```bash
python iris_flower_classification.py
```

This will:
- Export a pairplot of the dataset's feature relationships
- Split and scale the data
- Train and evaluate all three models (printing an accuracy table to the console)
- Save the model comparison and confusion matrix charts as `.png` files
- Save the trained pipeline to `iris_pipeline.pkl`
- Run a sample prediction on a custom flower measurement

### Example output

```
🌸 Iris Flower Classification System Started 🌸

📈 Data visualization charts exported successfully.
Model                | Validation Accuracy
------------------------------------------
LogisticRegression   | ...%
DecisionTree         | ...%
RandomForest          | ...%

🏆 Final Selected Model: ...
Top Validation Accuracy: ...%

--- Pipeline Manual Verification Case ---
Sample Flower Status Outcome: ...
```

## 🔮 Predicting a New Flower

You can reuse the trained pipeline (or the `IrisClassifier` class) to predict the species for new measurements:

```python
sample_flower = {
    'sepal_length': 6.1,
    'sepal_width': 2.8,
    'petal_length': 4.7,
    'petal_width': 1.2
}

predicted_species = classifier.predict_species(sample_flower, feature_names)
print(f"Predicted Species: {predicted_species.upper()}")
```

## 📊 Dataset

The dataset (`IRIS.csv`) contains 150 flower samples (50 per species) with the following columns:

| Column | Description |
|---|---|
| sepal_length | Sepal length (cm) |
| sepal_width | Sepal width (cm) |
| petal_length | Petal length (cm) |
| petal_width | Petal width (cm) |
| species | Flower species — target variable (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) |

## 📝 Notes

- The stratified split guarantees each species is proportionally represented in both training and test sets, which matters given the dataset's small, balanced size.
- Petal length and width tend to be the most discriminating features between species, as visible in the pairplot.
- This dataset is small and well-separated, so all three models typically achieve high accuracy — the comparison mainly highlights differences in decision boundaries rather than raw predictive power.

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `joblib`
