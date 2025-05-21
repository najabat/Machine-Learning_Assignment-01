# Machine Learning - Assignment 01

This is an assignment's solution covering two-parts of machine learning problems regarding regression (Part 1) and classification (Part 2).

## Project Structure

* `ML-Assignment01_Part01-Multilinear-Polynomial-Regression_Najabat.ipynb`: Part 1 - Regression analysis.
* `ML-Assignment01_Part02-NaiveBayes-KNN-DecisionTree_Najabat.ipynb`: Part 2 - Classification analysis.
* `README.md`: This overview.

## Part 1: Regression Analysis - Predicting Math Scores

### Objective
Predict student math scores using Multilinear and Polynomial Regression (Degree 2).

### Dataset
"Students Performance in Exams" dataset (downloaded from Kaggle). Features include student demographics, parental education, and test preparation.

### Key Steps
1. **Preprocessing:** Loaded data, handled missing values, encoded categorical features (One-Hot), and scaled numerical features (StandardScaler).
2. **EDA:** Analyzed feature correlations and relationships with the target.
3. **Modeling:** Split data (80/20), trained Multilinear and Polynomial (Degree 2) regression models.
4. **Evaluation:** Assessed models using RMSE, MAE, R²; visualized predictions; discussed overfitting.

## Part 2: Classification Analysis - Predicting Product Purchases

### Objective
Predict product purchases based on user age and estimated salary using Naive Bayes, KNN (k=3, 5, 7), and Decision Tree (Gini, Entropy) classifiers.

### Dataset
"Social Network Ads" dataset (downloaded from Kaggle). Features: 'Age', 'EstimatedSalary'. Target: 'Purchased'.

### Key Steps
1. **Preprocessing:** Loaded data, dropped 'User ID' & 'Gender', ensured 'Purchased' target was binary, and standardized 'Age' & 'EstimatedSalary'.
2. **EDA:** Visualized target distribution and feature relationships with the purchase status.
3. **Modeling:** Split data (75/25 with stratification), trained Naive Bayes, KNN (k=3, 5, 7), and Decision Tree (Gini, Entropy) models.
4. **Evaluation:** Assessed models using Accuracy, Precision, Recall, F1-Score, Classification Reports, and Confusion Matrices. Plotted decision boundaries. Compared model performance (detailed comparison available in the notebook).

## How to Run
1. Requires Python & Jupyter Notebook/Colab.
2. Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn (see notebook imports).
3. Data sets downloaded from Kaggle to be mounted (mounted by me already with provided files) in Google Drive for Colab.
4. Run notebook cells sequentially.

## Author
Najabat Ali Khan | DS & AI Bootcamp - Batch-11
