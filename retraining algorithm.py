import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

#preprocessing function
def preprocess_data(df, correlation_threshold=0.85):
    df = df.drop(columns=["Category", "Filename"])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    zero_cols = [col for col in numeric_cols if (df[col] == 0).all()]
    df = df.drop(columns=zero_cols)

    df["Class"] = df["Class"].apply(lambda x: 0 if x == "Benign" else 1)

    corrs = df.corr(numeric_only=True)["Class"].abs()
    leaky_cols = corrs[corrs > correlation_threshold].index.tolist()
    if "Class" in leaky_cols:
        leaky_cols.remove("Class")
    df = df.drop(columns=leaky_cols)

    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y

#train and save model
def train_model(X, y, model_path="ransomware_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Initial Training Complete:\n", classification_report(y_test, y_pred))

    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

#update and retrain model with new data
def update_model_with_new_data(new_data_path, existing_data_path, model_path="ransomware_model.pkl"):
    #load old and new data
    full_df = pd.read_csv(existing_data_path)
    new_df = pd.read_csv(new_data_path)

    #concatenate and deduplicate
    combined_df = pd.concat([full_df, new_df]).drop_duplicates()

    #check for new ransomware samples
    new_ransomware_count = (new_df["Class"] == "Malicious").sum()
    print(f"New ransomware samples: {new_ransomware_count}")

    if new_ransomware_count > 0:
        #preprocess and retrain
        X, y = preprocess_data(combined_df)
        train_model(X, y, model_path)
    else:
        print("No new ransomware found. Model not retrained.")

#prediction api
def predict_from_file(input_path, model_path="ransomware_model.pkl"):
    model = joblib.load(model_path)
    df = pd.read_csv(input_path)
    X, _ = preprocess_data(df)
    predictions = model.predict(X)
    return predictions

#initial training
df = pd.read_csv("MalMem2022.csv")
X, y = preprocess_data(df)
train_model(X, y)

#nimulate new ransomware arriving
update_model_with_new_data("new_ransomware_samples.csv", "MalMem2022.csv")

#predict from a new batch
preds = predict_from_file("unknown_samples.csv")
print(preds)

