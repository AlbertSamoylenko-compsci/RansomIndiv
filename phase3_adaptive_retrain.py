import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

LOG_FILE = "suspicious_log.csv"

#check if there's any logged data
if not os.path.exists(LOG_FILE):
    print("No log file found. Run phase 2 first.")
    exit()

#load logged data
df = pd.read_csv(LOG_FILE)

df.drop_duplicates(inplace=True)

#define features and target
X = df[["CPU%", "Mem%", "Threads", "Handles", "HasParent"]]
y = df["Label"]  # Currently all 1s, but you can manually label some 0s if needed

#split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#retrain model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#evaluate
y_pred = clf.predict(X_test)
print("\n--- Retrained Model Report ---")
print(classification_report(y_test, y_pred))

#overwrite previous model
joblib.dump(clf, "ransomware_detector.pkl")
print("\Model retrained and updated as 'ransomware_detector.pkl'")
