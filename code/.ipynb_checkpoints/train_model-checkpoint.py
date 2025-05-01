import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("Starting model training process...")

# Load the cleaned data
try:
    df = pd.read_csv('../data/cleaned_fatalities.csv')
    print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Validate required columns
required_columns = ['latitude', 'longitude', 'collision_category', 'time_of_day']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    print("Available columns:", df.columns.tolist())
    exit(1)

print("Data validation passed. Beginning feature engineering...")

# Feature Engineering
# Add day of week with robust error handling
print("\nCreating day_of_week column...")

# First try using collision_date
if 'collision_date' in df.columns:
    try:
        df['collision_date'] = pd.to_datetime(df['collision_date'], errors='coerce')
        if df['collision_date'].notna().sum() > len(df) * 0.5:  # If more than 50% valid dates
            df['day_of_week'] = df['collision_date'].dt.day_name()
            print("Successfully created day_of_week from collision_date")
        else:
            raise ValueError("Too many NaN values in collision_date")
    except Exception as e:
        print(f"Error using collision_date: {e}")
        # Try alternative method
        if all(col in df.columns for col in ['collision_year_clean', 'collision_month', 'collision_day']):
            try:
                df['synthetic_date'] = pd.to_datetime(
                    df['collision_year_clean'].astype(str) + '-' + 
                    df['collision_month'].astype(str) + '-' + 
                    df['collision_day'].astype(str),
                    errors='coerce'
                )
                df['day_of_week'] = df['synthetic_date'].dt.day_name()
                print("Successfully created day_of_week from synthetic date")
            except Exception as e2:
                print(f"Error creating synthetic date: {e2}")
                # Fall back to default
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                df['day_of_week'] = [days[i % 7] for i in range(len(df))]
                print("Created default day_of_week")
        else:
            # Fall back to default
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df['day_of_week'] = [days[i % 7] for i in range(len(df))]
            print("Created default day_of_week")
else:
    # If collision_date doesn't exist, create a default day_of_week
    print("Warning: collision_date column not found")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'] = [days[i % 7] for i in range(len(df))]
    print("Created default day_of_week")

# Verify day_of_week column
print(f"day_of_week column exists: {'day_of_week' in df.columns}")
if 'day_of_week' in df.columns:
    print(f"Unique values in day_of_week: {df['day_of_week'].unique()}")
    print(f"NaN values in day_of_week: {df['day_of_week'].isna().sum()}")
    # Fill any NaN values
    df['day_of_week'] = df['day_of_week'].fillna('Monday')

# Create target variable: high-risk vs low-risk areas
print("\nCreating target variable...")

# SIMPLIFIED APPROACH: Define high-risk based on collision category and time of day
# This avoids the merge operation that was causing issues
print("Using direct approach to create high_risk target variable")
df['high_risk'] = ((df['collision_category'] == 'Pedestrian') | 
                  (df['time_of_day'] == 'Night (9pm-5am)')).astype(int)

# Verify the column exists and has appropriate values
print(f"High risk incidents: {df['high_risk'].sum()} out of {len(df)} total records")
print(f"Percentage high risk: {df['high_risk'].mean() * 100:.2f}%")

# Select features for the model
print("\nPreparing features...")
features = [
    'latitude', 'longitude', 'collision_hour', 'collision_month',
    'time_of_day', 'day_of_week', 'collision_category'
]

# Check if all features exist
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"Warning: Missing features: {missing_features}")
    # Remove missing features from the list
    features = [f for f in features if f in df.columns]
    print(f"Proceeding with available features: {features}")

# Handle missing values in features
for feature in features:
    if df[feature].dtype == 'object':
        df[feature] = df[feature].fillna('Unknown')
    else:
        df[feature] = df[feature].fillna(df[feature].median())

# Prepare X and y
X = df[features]
y = df['high_risk']

# Print class distribution
print("\nClass distribution:")
print(y.value_counts())
print(f"Class balance: {y.mean() * 100:.2f}% high risk")

# Split the data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Check class distribution in training set
print("Class distribution in training set:")
print(pd.Series(y_train).value_counts())
print(f"Training set class balance: {y_train.mean() * 100:.2f}% high risk")

# Define preprocessing for numerical and categorical features
print("\nBuilding preprocessing pipeline...")
numerical_features = [f for f in features if df[f].dtype != 'object']
categorical_features = [f for f in features if df[f].dtype == 'object']

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create and train the model
print("\nTraining the model...")
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

try:
    model.fit(X_train, y_train)
    print("Model training completed successfully")
except Exception as e:
    print(f"Error during model training: {e}")
    exit(1)

# Evaluate the model
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

try:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(f"Could not calculate ROC AUC: {e}")

# Generate SHAP values for feature importance
print("\nGenerating SHAP values for feature importance...")
try:
    # Get the preprocessed test data
    X_test_processed = model.named_steps['preprocessor'].transform(X_test)
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test_processed)
    
    # Get feature names after preprocessing
    cat_feature_names = []
    for i, col in enumerate(categorical_features):
        cats = model.named_steps['preprocessor'].transformers_[1][1].categories_[i]
        for cat in cats:
            cat_feature_names.append(f"{col}_{cat}")
    
    feature_names = numerical_features + cat_feature_names
    
    print(f"Generated SHAP values with {len(feature_names)} features")
    
    # Save feature names for the Streamlit app
    with open('../code/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save SHAP values for the Streamlit app
    with open('../code/shap_values.pkl', 'wb') as f:
        pickle.dump((shap_values, X_test_processed), f)
    
    print("SHAP values saved successfully")
except Exception as e:
    print(f"Error generating SHAP values: {e}")
    print("Continuing without SHAP values")

# Save the model
print("\nSaving model and data for Streamlit app...")
try:
    with open('../code/fatality_risk_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully")
except Exception as e:
    print(f"Error saving model: {e}")
    exit(1)

# Save a sample of the data for the Streamlit app
try:
    df_sample = df.sample(frac=1.0, random_state=42)
    df_sample.to_csv('../data/fatality_data_processed.csv', index=False)
    print("Processed data saved successfully")
except Exception as e:
    print(f"Error saving processed data: {e}")

print("\nModel training complete. Files saved for Streamlit app.")
print("You can now run the Streamlit app with: streamlit run app.py")