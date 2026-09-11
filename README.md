# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived
the Titanic disaster using classification algorithms.

## Project Overview

This project compares multiple machine learning classification models
and evaluates their performance using accuracy and a confusion matrix.
The models considered are:

-   Logistic Regression
-   Decision Tree
-   Random Forest

The goal is to identify the model that provides the best prediction
performance for Titanic passenger survival.

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Joblib

The project dependencies are listed in `requirements.txt`.

## Model Performance

The accuracy benchmark shows the following results:

  Model                   Accuracy
  --------------------- ----------
  Logistic Regression        79.3%
  Decision Tree              76.0%
  Random Forest              78.7%

Based on the accuracy comparison, **Logistic Regression** is the
best-performing model among the three tested models, with an accuracy of
**79.3%**.

![Model Accuracy Benchmark](model_comparison.png)

## Confusion Matrix

The confusion matrix for the best-performing model contains the
following results:

                          Predicted Deceased   Predicted Survived
  --------------------- -------------------- --------------------
  **Actual Deceased**                     95                   15
  **Actual Survived**                     22                   47

This means:

-   **95** deceased passengers were correctly predicted as deceased.
-   **15** deceased passengers were predicted as survived.
-   **22** survived passengers were predicted as deceased.
-   **47** survived passengers were correctly predicted as survived.

![Confusion Matrix](titanic_confusion_matrix.png)

## Project Workflow

1.  Load the Titanic dataset.
2.  Perform data preprocessing and prepare the required features.
3.  Split the data into training and testing sets.
4.  Train classification models.
5.  Evaluate each model using accuracy.
6.  Compare model performance.
7.  Select the best-performing model.
8.  Visualize the model comparison and confusion matrix.
9.  Save the trained model using Joblib when required.

## Evaluation

Model performance is compared using classification accuracy. The
confusion matrix is also used to understand correct and incorrect
predictions for the two classes:

-   Deceased
-   Survived

## Installation

Clone or download the project and install the required Python packages:

``` bash
pip install -r requirements.txt
```

The current dependency list includes Pandas, NumPy, Matplotlib, Seaborn,
Scikit-learn, and Joblib.

## How to Run

1.  Make sure Python is installed.
2.  Install the dependencies using `requirements.txt`.
3.  Place the Titanic dataset and project files in the appropriate
    project directory.
4.  Run the Python machine learning script/notebook.
5.  Review the model accuracy comparison and confusion matrix.

## Project Structure

``` text
Titanic-Survival-Prediction/
│
├── dataset/
│   └── titanic.csv
│
├── model_comparison.png
├── titanic_confusion_matrix.png
├── requirements.txt
├── train_model.py
└── README.md
```

> The exact dataset and Python script filenames may differ depending on
> the project implementation.

## Results

The experiment shows that Logistic Regression achieved the highest
accuracy among the evaluated models at **79.3%**. Decision Tree achieved
**76.0%**, while Random Forest achieved **78.7%**.

The confusion matrix provides additional insight into how the selected
model classified deceased and survived passengers.

## Future Enhancements

-   Perform more detailed feature engineering.
-   Tune model hyperparameters.
-   Compare additional classification algorithms.
-   Add precision, recall, and F1-score evaluation.
-   Build a simple web interface for making passenger survival
    predictions.
-   Deploy the trained model as an application.

## Conclusion

The Titanic Survival Prediction project demonstrates how machine
learning classification algorithms can be used to predict passenger
survival. Among the tested models, Logistic Regression produced the
highest accuracy in the provided benchmark. The confusion matrix further
helps evaluate the model's prediction behavior across the two survival
classes.

## License

This project is intended for educational and learning purposes.
