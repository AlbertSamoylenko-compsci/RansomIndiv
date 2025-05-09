import psutil
import pandas as pd
import joblib
import time
from datetime import datetime
import os

#load the trained model
model = joblib.load("ransomware_detector.pkl")

#path to log suspicious detections
LOG_FILE = "suspicious_log.csv"

#ensure log file exists with header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,PID,Name,CPU%,Mem%,Threads,Handles,HasParent,Label\n")

print("Monitoring processes in real time...\n")

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
                print(f"Suspicious process detected: PID={pid}, Name={name}")

                #append to log with label = 1 (initially marked as ransomware)
                timestamp = datetime.now().isoformat()
                with open(LOG_FILE, "a") as f:
                    f.write(f"{timestamp},{pid},{name},{cpu},{mem},{threads},{handles},{has_parent},1\n")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    time.sleep(5)

