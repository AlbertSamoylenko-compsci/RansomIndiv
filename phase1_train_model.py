import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

#load MalMem-2022 dataset
df = pd.read_csv("MalMem2022.csv")

#convert labels to binary: Ransomware = 1, Benign = 0
df["Label"] = df["Category"].apply(lambda x: 0 if x == "Benign" else 1)

#simulate psutil-style features (since MalMem doesn't have them)
np.random.seed(42)

df["CPU%"] = np.random.uniform(0, 100, size=len(df))
df["Mem%"] = np.random.uniform(0, 30, size=len(df))
df["Threads"] = np.random.randint(1, 100, size=len(df))
df["Handles"] = np.random.randint(10, 500, size=len(df))
df["HasParent"] = np.random.choice([0, 1], size=len(df))

#select only synthetic features
X = df[["CPU%", "Mem%", "Threads", "Handles", "HasParent"]]
y = df["Label"]

#train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#evaluate
y_pred = clf.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

#save model
joblib.dump(clf, "ransomware_detector.pkl")
print("Model trained and saved as 'ransomware_detector.pkl'")

