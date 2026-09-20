# 🎬 Movie Rating Prediction with Python

A machine learning project that predicts IMDb ratings for Indian movies based on features like genre, director, cast, release year, and duration. Built as part of a data science internship task.

## 📌 Overview

This project uses the **IMDb India Movies** dataset to build a regression pipeline that predicts a movie's IMDb rating. It covers the full workflow: data cleaning, feature engineering (target encoding for categorical fields), model training, evaluation, and persistence of the final pipeline for reuse.

## 📂 Project Structure

```
Task2/
├── movie_rating_prediction.py       # Main script (cleaning, training, evaluation, prediction)
├── IMDb_Movies _India.csv           # Raw dataset (~15.5K Indian movies)
├── movie_rating_pipeline.pkl        # Saved model + scaler + encodings (joblib)
├── movie_rating_distribution.png    # Distribution of IMDb ratings
├── regression_model_comparison.png  # R² benchmark across models
├── actual_vs_predicted.png          # Actual vs. predicted ratings scatter plot
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🧠 Approach

1. **Data Cleaning**
   - Loads the CSV with `latin-1` encoding to handle non-ASCII characters in Indian names.
   - Drops rows with a missing `Rating` (the prediction target).
   - Extracts numeric values from messy `Duration` (e.g. `"109 min"` → `109`) and `Year` (e.g. `"(2019)"` → `2019`) columns, filling any remaining gaps with the median.
   - Fills missing categorical fields (`Genre`, `Director`, `Actor 1/2/3`) with `"Unknown"`.

2. **Feature Engineering**
   - Splits data into train/test sets **before** encoding to avoid data leakage.
   - Applies **target (mean) encoding** to categorical columns using statistics from the training set only.
   - Scales all features with `StandardScaler`.

3. **Model Training & Evaluation**
   Three regression models are trained and compared:
   - Linear Regression
   - Gradient Boosting Regressor
   - Random Forest Regressor

   Each is evaluated using **MAE**, **RMSE**, and **R² Score**, and the best-performing model (by R²) is automatically selected.

4. **Visualization**
   - `movie_rating_distribution.png` — distribution of ratings in the dataset.
   - `regression_model_comparison.png` — R² comparison across the three models.
   - `actual_vs_predicted.png` — scatter plot of predicted vs. actual ratings for the best model.

5. **Persistence & Inference**
   - The trained model, scaler, encoding maps, and global mean are saved to `movie_rating_pipeline.pkl` via `joblib`.
   - `predict_rating()` demonstrates predicting a rating for a custom, unseen movie input.

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

## 🚀 Usage

Make sure `IMDb_Movies _India.csv` is in the same directory, then run:

```bash
python movie_rating_prediction.py
```

This will:
- Clean and preprocess the dataset
- Train and evaluate all three models (printing a metrics table to the console)
- Save evaluation charts as `.png` files
- Save the trained pipeline to `movie_rating_pipeline.pkl`
- Run a sample prediction on a custom movie entry

### Example output

```
🎬 Movie Rating Prediction System Started 🎬

📈 Data visualization charts exported successfully.
Model                | MAE      | RMSE     | R² Score
----------------------------------------------------
LinearRegression     | ...      | ...      | ...
GradientBoosting     | ...      | ...      | ...
RandomForest         | ...      | ...      | ...

🏆 Final Selected Model: LinearRegression
Top Validation R² Score: ...

--- Pipeline Manual Verification Case ---
Predicted IMDb Score for custom case: X.X/10
```

## 🔮 Predicting a New Movie

You can reuse the saved pipeline (or the `MovieRatingPredictor` class) to predict ratings for new movies:

```python
sample_movie = {
    'Year': 2026.0,
    'Duration': 140.0,
    'Genre': 'Action, Drama',
    'Director': 'S.S. Rajamouli',
    'Actor 1': 'Prabhas',
    'Actor 2': 'Rana Daggubati',
    'Actor 3': 'Anushka Shetty'
}

predicted_score = predictor.predict_rating(sample_movie, feature_names)
print(f"Predicted IMDb Score: {predicted_score}/10")
```

## 📊 Dataset

The dataset (`IMDb_Movies _India.csv`) contains ~15,500 Indian movie entries with the following columns:

| Column | Description |
|---|---|
| Name | Movie title |
| Year | Release year |
| Duration | Runtime (minutes) |
| Genre | Genre(s) |
| Rating | IMDb rating (target variable) |
| Votes | Number of votes |
| Director | Director name |
| Actor 1 / 2 / 3 | Lead cast members |

## 📝 Notes

- Target encoding is fit strictly on the training split to prevent leakage into the test set.
- Predictions are clipped to the valid IMDb range of `1.0`–`10.0`.
- Model performance is inherently limited by the noisiness of crowd-sourced ratings and the modest signal in categorical metadata alone — this project is primarily a demonstration of a clean, leakage-safe ML pipeline rather than a production-grade rating predictor.

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `joblib`
