import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("heart.csv")
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

print("\nBasic Decision Tree")
train_accuracy = accuracy_score(y_train, dt_train_pred)
test_accuracy = accuracy_score(y_test, dt_test_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Accuracy Gap: {train_accuracy - test_accuracy:.4f}")
if train_accuracy - test_accuracy > 0.05:
    print("The model may be overfitting.")
else:
    print("No large training-testing accuracy gap detected.")

plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png", bbox_inches="tight")
plt.show()

controlled_dt = DecisionTreeClassifier(
    max_depth=5, random_state=42
)
controlled_dt.fit(X_train, y_train)
print("\nControlled Decision Tree")
print("Training Accuracy:",
      accuracy_score(y_train, controlled_dt.predict(X_train)))
print("Testing Accuracy:",
      accuracy_score(y_test, controlled_dt.predict(X_test)))

dt_params = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_
print("\nTuned Decision Tree")
print("Best Parameters:", dt_grid.best_params_)
print("Best CV Accuracy:", dt_grid.best_score_)
print("Test Accuracy:",
      accuracy_score(y_test, best_dt.predict(X_test)))

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
rf_model.fit(X_train, y_train)
print("\nBasic Random Forest")
print("Training Accuracy:",
      accuracy_score(y_train, rf_model.predict(X_train)))
print("Testing Accuracy:",
      accuracy_score(y_test, rf_model.predict(X_test)))

rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("\nTuned Random Forest")
print("Best Parameters:", rf_grid.best_params_)
print("Best CV Accuracy:", rf_grid.best_score_)
print("Test Accuracy:",
      accuracy_score(y_test, best_rf.predict(X_test)))

print("\nTuned Random Forest Evaluation")
rf_predictions = best_rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

models = {
    "Basic Decision Tree": dt_model,
    "Controlled Decision Tree": controlled_dt,
    "Tuned Decision Tree": best_dt,
    "Basic Random Forest": rf_model,
    "Tuned Random Forest": best_rf
}
results = []
for name, model in models.items():
    train_accuracy = accuracy_score(
        y_train, model.predict(X_train)
    )
    test_accuracy = accuracy_score(
        y_test, model.predict(X_test)
    )
    results.append({
        "Model": name,
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy
    })
results_df = pd.DataFrame(results)
print("\nModel Comparison")
print(results_df.round(4).to_string(index=False))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
})
importance = importance.sort_values(
    by="Importance", ascending=False
)
print("\nFeature Importance")
print(importance.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.show()

dt_scores = cross_val_score(
    best_dt, X_train, y_train,
    cv=5, scoring="accuracy", n_jobs=-1
)
rf_scores = cross_val_score(
    best_rf, X_train, y_train,
    cv=5, scoring="accuracy", n_jobs=-1
)
print("\n5-Fold Cross-Validation")
print("Tuned Decision Tree Scores:", dt_scores.round(4))
print("Decision Tree Mean CV Accuracy:", round(dt_scores.mean(), 4))
print("\nTuned Random Forest Scores:", rf_scores.round(4))
print("Random Forest Mean CV Accuracy:", round(rf_scores.mean(), 4))

plt.figure(figsize=(20, 10))
plot_tree(
    best_dt,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)
plt.title("Tuned Decision Tree Visualization")
plt.savefig("tuned_decision_tree.png", bbox_inches="tight")
plt.show()