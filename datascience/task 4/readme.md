
# 📈 Sales Prediction Using Python

A machine learning project that predicts product sales based on advertising spend across TV, Radio, and Newspaper channels. Built as part of a data science internship task.

## 📌 Overview

This project uses the classic **Advertising dataset** to build a regression pipeline that estimates sales output from a given marketing budget allocation. It covers the full workflow: exploratory correlation analysis, preprocessing, model benchmarking, evaluation, and persistence of the final pipeline for reuse.

## 📂 Project Structure

```
Task4_Sales_Prediction_Using_Python/
├── sales_prediction_using_python.py   # Main script (analysis, training, evaluation, prediction)
├── advertising.csv                    # Raw dataset (200 records)
├── sales_prediction_pipeline.pkl      # Saved model + scaler (joblib)
├── sales_correlation_matrix.png       # Correlation heatmap of channels vs sales
├── regression_model_comparison.png    # R² benchmark across models
├── actual_vs_predicted_sales.png      # Actual vs predicted sales scatter plot
├── requirements.txt                   # Python dependencies
└── README.md
```

## 🧠 Approach

1. **Data Loading & Cleanup**
   - Loads `advertising.csv` and drops any stray index column (e.g. `Unnamed: 0`) if present.
   - Features used: `TV`, `Radio`, `Newspaper` ad spend. Target: `Sales`.

2. **Exploratory Analysis**
   - Computes and visualizes a correlation matrix between the three advertising channels and sales, highlighting which channels most strongly drive sales.

3. **Preprocessing**
   - Performs an 80/20 train/test split.
   - Scales all features with `StandardScaler` for stable optimization across models.

4. **Model Training & Evaluation**
   Three regression models are trained and compared:
   - Linear Regression
   - Gradient Boosting Regressor
   - Random Forest Regressor

   Each is evaluated using **MAE**, **RMSE**, and **R² Score**, and the best-performing model (by R²) is automatically selected.

5. **Visualization**
   - `sales_correlation_matrix.png` — heatmap of correlations between ad spend channels and sales.
   - `regression_model_comparison.png` — R² comparison across the three models.
   - `actual_vs_predicted_sales.png` — scatter plot of predicted vs. actual sales for the best model.

6. **Persistence & Inference**
   - The trained model and scaler are saved to `sales_prediction_pipeline.pkl` via `joblib`.
   - `predict_sales()` demonstrates estimating sales for a custom marketing budget mix.

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

## 🚀 Usage

> ⚠️ **Before running:** the script currently points to a hardcoded local path for the dataset:
> ```python
> FILE_NAME = r"C:\Users\yashw\Yashu\Internship\DATA_SCIENCE\Task4\advertising.csv"
> ```
> Update `FILE_NAME` to `"advertising.csv"` (or the correct path on your machine) before running, assuming `advertising.csv` is in the same folder as the script.

Then run:

```bash
python sales_prediction_using_python.py
```

This will:
- Export a correlation heatmap of advertising channels vs. sales
- Split and scale the data
- Train and evaluate all three models (printing an MAE/RMSE/R² table to the console)
- Save the model comparison and actual-vs-predicted charts as `.png` files
- Save the trained pipeline to `sales_prediction_pipeline.pkl`
- Run a sample prediction on a custom marketing budget mix

### Example output

```
📈 Sales Prediction System Initialization Started 📈

📊 Data correlation graphs exported successfully.
Model                | MAE      | RMSE     | R² Score
----------------------------------------------------
LinearRegression     | ...      | ...      | ...
GradientBoosting     | ...      | ...      | ...
RandomForest         | ...      | ...      | ...

🏆 Final Selected Model: ...
Top Validation R² Score: ...

--- Pipeline Manual Verification Case ---
Estimated Sales Output: ... units
```

## 🔮 Predicting Sales for a New Budget

You can reuse the trained pipeline (or the `SalesPredictor` class) to estimate sales for a new marketing budget allocation:

```python
sample_marketing_mix = {
    'TV': 250.0,
    'Radio': 35.0,
    'Newspaper': 15.0
}

predicted_revenue = predictor.predict_sales(sample_marketing_mix, feature_names)
print(f"Estimated Sales Output: {predicted_revenue} units")
```

## 📊 Dataset

The dataset (`advertising.csv`) contains 200 records with the following columns:

| Column | Description |
|---|---|
| TV | Advertising spend on TV (in thousands) |
| Radio | Advertising spend on Radio (in thousands) |
| Newspaper | Advertising spend on Newspaper (in thousands) |
| Sales | Resulting product sales — target variable |

## 📝 Notes

- TV spend is typically the strongest predictor of sales in this dataset, as reflected in the correlation heatmap.
- Predicted sales are clipped at a minimum of `0.0` to avoid nonsensical negative output.
- With only 200 records and 3 features, tree-based ensembles can overfit slightly relative to linear regression — the R² comparison chart makes this trade-off visible.

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `joblib`
