# Decision Trees and Random Forests for Heart Disease Classification

## Project Overview

The objective of this project is to understand and implement **Decision Tree and Random Forest algorithms** for a binary classification problem.The project uses a heart disease dataset containing medical attributes that can be used to predict whether a person has heart disease.

The target variable is `target`, which represents two classes:
1. **No Disease (0)** – the person is classified as not having heart disease.
2. **Disease (1)** – the person is classified as having heart disease.
Two machine learning algorithms, Decision Tree Classifier and Random Forest Classifier, are trained and evaluated. The project also explores overfitting, hyperparameter tuning, feature importance, and cross-validation to compare model performance.

## Objectives

- Import and explore the heart disease dataset.
- Separate input features from the target variable.
- Split the dataset into training and testing sets.
- Implement a **Decision Tree Classifier**.
- Visualize the Decision Tree.
- Analyze overfitting by comparing training and testing accuracy.
- Control tree complexity using the `max_depth` parameter.
- Implement a **Random Forest Classifier**.
- Compare the performance of both algorithms.
- Perform hyperparameter tuning using `GridSearchCV`.
- Identify important features using Random Forest feature importance.
- Evaluate models using 5-fold cross-validation.
- Generate classification reports and confusion matrices.

## Technologies Used

- **Python** – programming language
- **Pandas** – data loading, manipulation, and result comparison
- **Matplotlib** – visualization of decision trees and feature importance
- **Scikit-learn** – model training, hyperparameter tuning, cross-validation, and evaluation

## Dataset

The project uses a heart disease dataset stored in `heart.csv`.
The dataset contains medical attributes used to predict the presence or absence of heart disease.

### Target Variable

The target column is `target`.

| Value | Meaning |
|---|---|
| `0` | No Disease |
| `1` | Disease |
The dataset's target labels are used directly for classification.

### Features

The dataset includes medical attributes such as:

- `age` – age of the individual
- `sex` – sex category encoded in the dataset
- `cp` – chest pain type
- `trestbps` – resting blood pressure
- `chol` – serum cholesterol
- `fbs` – fasting blood sugar indicator
- `restecg` – resting electrocardiographic results
- `thalach` – maximum heart rate achieved
- `exang` – exercise-induced angina indicator
- `oldpeak` – ST depression induced by exercise
- `slope` – slope of the peak exercise ST segment
- `ca` – number of major vessels represented in the dataset
- `thal` – thalassemia-related category

The exact feature names depend on the columns present in `heart.csv`.

## Working

### Step 1: Data Loading and Exploration

- Imported Pandas and Matplotlib.
- Loaded the heart disease dataset.
- Displayed the dataset shape and first five rows.
- Separated the input features and target variable.

### Step 2: Train-Test Split

The dataset was divided into training and testing sets using `train_test_split()`.
- **80% Training Data** – used to train the models.
- **20% Testing Data** – used to evaluate model performance on held-out data.
The parameter `stratify=y` maintains the target class proportions across the two sets.
The `random_state=42` parameter ensures reproducibility of the split.

### Step 3: Decision Tree Classification

A Decision Tree Classifier was trained using the training dataset.
The model makes predictions by splitting data into branches according to feature values and learned decision rules. Training and testing accuracy were calculated to evaluate the model's performance and identify potential overfitting.

### Step 4: Decision Tree Visualization

The trained Decision Tree was visualized using Scikit-learn's `plot_tree()` function.
The visualization displays:
- Feature-based splitting conditions
- Decision nodes and leaf nodes
- Predicted classes
- Sample distributions at nodes
The visualization is limited to the first three levels to make the tree easier to inspect.
The generated image is saved as `decision_tree.png`.

### Step 5: Overfitting Analysis

Overfitting occurs when a model learns the training data too closely and performs less effectively on unseen data. To investigate overfitting, the training and testing accuracy of the Decision Tree were compared.
The accuracy gap was calculated as:
`Accuracy Gap = Training Accuracy - Testing Accuracy`

A large positive gap can indicate potential overfitting.

### Step 6: Controlling Tree Depth

A second Decision Tree was trained with `max_depth=5`.
Limiting tree depth reduces model complexity and can help prevent overfitting. The controlled model's training and testing accuracy were compared with those of the basic Decision Tree.

### Step 7: Decision Tree Hyperparameter Tuning

`GridSearchCV` was used to identify suitable Decision Tree hyperparameters.
The following parameters were tested:
- `max_depth`: `[3, 5, 10, None]`
- `min_samples_split`: `[2, 5, 10]`
- `min_samples_leaf`: `[1, 2, 4]`

Five-fold cross-validation was used to evaluate the parameter combinations. The configuration with the highest mean cross-validation accuracy was selected using `best_params_`.
The resulting model was evaluated on the held-out test dataset.

### Step 8: Random Forest Classification

A Random Forest Classifier was trained using 100 decision trees.
Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.
It can improve generalization by combining trees trained with randomized samples and feature selection. Training and testing accuracy were calculated for comparison with the basic Decision Tree.

### Step 9: Random Forest Hyperparameter Tuning

`GridSearchCV` was used to tune the Random Forest model.
The following hyperparameters were tested:
- `n_estimators`: `[50, 100, 200]`
- `max_depth`: `[5, 10, None]`
- `min_samples_split`: `[2, 5]`
- `min_samples_leaf`: `[1, 2]`

Five-fold cross-validation was used to compare the parameter combinations. The best configuration was selected based on mean cross-validation accuracy and evaluated on the test dataset.

### Step 10: Model Comparison

The performance of five models was compared:
1. Basic Decision Tree
2. Controlled Decision Tree
3. Tuned Decision Tree
4. Basic Random Forest
5. Tuned Random Forest

Training and testing accuracy were displayed in a comparison table. This comparison helps demonstrate how tree-depth control, ensemble learning, and hyperparameter tuning affect model performance.

### Step 11: Feature Importance

Feature importance was extracted from the tuned Random.
The importance values indicate the relative contribution of individual features to the model's predictions. The values were sorted in descending order to identify the most influential features.
A horizontal bar chart was created to visualize the feature importance values and saved as `feature_importance.png`.

### Step 12: Cross-Validation

Five-fold cross-validation was performed for the tuned Decision Tree and Random Forest models.
The training dataset was divided into five folds. Each fold was used for validation while the remaining folds were used for training. The individual scores and mean cross-validation accuracy were printed for both models.
This provides information about performance consistency across different data splits.

### Step 13: Tuned Decision Tree Visualization

The tuned Decision Tree was visualized using `plot_tree()`.
The first three levels were displayed for readability, and the resulting image was saved as `tuned_decision_tree.png`.

## Decision Tree Classifier

A Decision Tree is a supervised machine learning algorithm that uses a tree-like structure to make predictions. It repeatedly splits the dataset based on feature values until it reaches a decision or leaf node.
The algorithm can use criteria such as Gini impurity or entropy to determine suitable splits. In this project, the Decision Tree was used as a baseline model and then controlled using tree depth and hyperparameter tuning.

## Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. For classification, the trees' predictions are combined to determine the final class.
Random Forest can reduce the instability of a single decision tree and often generalizes better, although the actual performance depends on the dataset and hyperparameters. In this project, Random Forest was trained with 100 trees initially and subsequently tuned using `GridSearchCV`.

## Hyperparameter Tuning

Hyperparameter tuning is the process of testing different model configurations to identify one that performs well on validation data. In this project, `GridSearchCV` was used for both algorithms.
The best parameter combinations were selected using five-fold cross-validation and accuracy as the scoring metric. The selected models were then evaluated using the held-out test dataset.

## Model Evaluation

The models were evaluated using training accuracy, testing accuracy, classification reports, confusion matrices, and cross-validation.

### Training Accuracy

Training accuracy measures the proportion of correct predictions on the training dataset.

### Testing Accuracy

Testing accuracy measures the proportion of correct predictions on the held-out test dataset.

### Classification Report

The classification report provides:
- **Precision** – proportion of positive predictions that are correct.
- **Recall** – proportion of actual positive cases correctly identified.
- **F1-Score** – harmonic mean of precision and recall.
- **Support** – number of actual samples in each class.

### Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for each class. It helps identify true positives, true negatives, false positives, and false negatives.

### Cross-Validation Accuracy

Five-fold cross-validation evaluates model performance across multiple training and validation splits. The mean score provides a summary of the model's validation accuracy, while individual scores show how performance varies across folds.

## Outputs

The project generates or displays:
- `decision_tree.png` – basic Decision Tree visualization
- `tuned_decision_tree.png` – tuned Decision Tree visualization
- `feature_importance.png` – Random Forest feature importance chart
- Training and testing accuracy for all five models
- Overfitting analysis through the accuracy gap
- Best hyperparameters for both algorithms
- Best cross-validation scores from GridSearchCV
- Classification report for the tuned Random Forest
- Confusion matrix for the tuned Random Forest
- Model comparison table
- Feature importance values
- Five-fold cross-validation scores and mean accuracy