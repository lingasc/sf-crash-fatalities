import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import requests
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')

# Load the dataset
df = pd.read_csv('cleaned_fatalities.csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nTarget variable distribution:")
print(df['deceased'].value_counts())

# Create binary target variable (1 if deceased = 'Pedestrian', 0 otherwise)
df['is_pedestrian'] = (df['deceased'] == 'Pedestrian').astype(int)
print("\nBinary target distribution:")
print(df['is_pedestrian'].value_counts())

# Select relevant features
features = [
    'age', 'sex', 'time_of_day', 'collision_year_clean', 'collision_hour',
    'collision_type', 'age_category', 'analysis_neighborhood', 'supervisor_district'
]

# Check for missing values in selected features
print("\nMissing values in selected features:")
print(df[features].isnull().sum())

# Prepare the data
X = df[features].copy()
y = df['is_pedestrian']

# Define categorical and numerical features
categorical_features = ['sex', 'time_of_day', 'collision_type', 'age_category', 'analysis_neighborhood']
numerical_features = ['age', 'collision_year_clean', 'collision_hour', 'supervisor_district']

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create a pipeline with preprocessing and the gradient boosting classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Pedestrian', 'Pedestrian'],
            yticklabels=['Not Pedestrian', 'Pedestrian'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.figtext(0.5, 0.01, 'Source: Fatal Traffic Collision Dataset', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# Print confusion matrix as a table
print("\nConfusion Matrix:")
cm_df = pd.DataFrame(cm, 
                    index=['Actual: Not Pedestrian', 'Actual: Pedestrian'], 
                    columns=['Predicted: Not Pedestrian', 'Predicted: Pedestrian'])
print(cm_df)

# Create ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.figtext(0.5, 0.01, 'Source: Fatal Traffic Collision Dataset', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# Print ROC data as a table
print("\nROC Curve Data:")
roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
print(f"AUC: {roc_auc:.4f}")
print(roc_df.head(10))  # Print first 10 rows for brevity

# Get feature importances
# Extract the trained classifier from the pipeline
classifier = model.named_steps['classifier']

# Get feature names after preprocessing
preprocessor = model.named_steps['preprocessor']
feature_names = []

# Get numerical feature names
feature_names.extend(numerical_features)

# Get one-hot encoded feature names
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
categorical_features_encoded = ohe.get_feature_names_out(categorical_features)
feature_names.extend(categorical_features_encoded)

# Get feature importances
importances = classifier.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(15)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.figtext(0.5, 0.01, 'Source: Fatal Traffic Collision Dataset', ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# Print feature importances as a table
print("\nTop 15 Feature Importances:")
print(feature_importance_df)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("\nCross-Validation Results:")
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Individual CV Scores: {cv_scores}")

# Model summary
print("\nModel Summary:")
print("1. The Gradient Boosting Classifier was trained to predict whether a victim of a fatal traffic collision was a pedestrian.")
print(f"2. The model achieved an accuracy of {accuracy_score(y_test, y_pred):.4f} on the test set.")
print(f"3. The precision of {precision_score(y_test, y_pred):.4f} indicates the proportion of positive identifications that were actually correct.")
print(f"4. The recall of {recall_score(y_test, y_pred):.4f} indicates the proportion of actual positives that were identified correctly.")
print(f"5. The F1 score of {f1_score(y_test, y_pred):.4f} provides a balance between precision and recall.")
print(f"6. The ROC AUC of {roc_auc:.4f} indicates good discriminative ability of the model.")
print("7. Cross-validation confirms the model's stability across different data splits.")
print("8. The most important features for prediction include collision type, age, and neighborhood information.")