import psutil
import pandas as pd
import joblib
import time
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_FILE = "ransomware_detector.pkl"
LOG_FILE = "suspicious_log.csv"
RETRAIN_THRESHOLD = 20  #retrain after 20 new suspicious entries

#initialize or load model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    print("Model not found. Run initial training first.")
    exit()

#ensure log file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,PID,Name,CPU%,Mem%,Threads,Handles,HasParent,Label\n")

def log_suspicious(pid, name, cpu, mem, threads, handles, has_parent):
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp},{pid},{name},{cpu},{mem},{threads},{handles},{has_parent},1\n")

def retrain_model():
    print("\n🔄 Retraining model on new data...")
    df = pd.read_csv(LOG_FILE)
    df.drop_duplicates(inplace=True)

    X = df[["CPU%", "Mem%", "Threads", "Handles", "HasParent"]]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nRetrain Performance:")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_FILE)
    print("Model updated and saved.")

    return clf

# Keep track of entries for retraining
entries_since_last_train = 0

print("Ransomware Guardian started. Monitoring and adapting...\n")

while True:
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']):
        try:
            pid = proc.info['pid']
            name = proc.info['name']
            cpu = proc.info['cpu_percent']
            mem = proc.info['memory_percent']
            threads = proc.info['num_threads']
            handles = 100  
            has_parent = 1 if psutil.Process(pid).ppid() != 0 else 0

            features = pd.DataFrame([{
                "CPU%": cpu,
                "Mem%": mem,
                "Threads": threads,
                "Handles": handles,
                "HasParent": has_parent
            }])

            pred = model.predict(features)[0]

            if pred == 1:
                print(f"⚠️  Suspicious process detected: PID={pid}, Name={name}")
                log_suspicious(pid, name, cpu, mem, threads, handles, has_parent)
                entries_since_last_train += 1

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    #retrain after threshold
    if entries_since_last_train >= RETRAIN_THRESHOLD:
        model = retrain_model()
        entries_since_last_train = 0

    time.sleep(5)
