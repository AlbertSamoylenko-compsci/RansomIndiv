import pandas as pd
import numpy as np

#simulate new ransomware-like samples
def create_mock_ransomware_samples(output_path="new_ransomware_samples.csv", n_samples=10):
    data = {
        "Category": ["Ransomware"] * n_samples,
        "Filename": [f"ransom_sample_{i}.exe" for i in range(n_samples)],
        "dlllist.avg_dlls_per_proc": np.random.randint(100, 150, size=n_samples),
        "handles.nevent": np.random.randint(50, 100, size=n_samples),
        "handles.nthread": np.random.randint(100, 200, size=n_samples),
        "handles.nmutant": np.random.randint(20, 50, size=n_samples),
        "dlllist.ndlls": np.random.randint(300, 400, size=n_samples),
        "handles.nsection": np.random.randint(150, 250, size=n_samples),
        "pslist.avg_threads": np.random.uniform(10.0, 25.0, size=n_samples),
        "ldrmodules.not_in_load": np.random.randint(10, 30, size=n_samples),
        "ldrmodules.not_in_mem": np.random.randint(10, 30, size=n_samples),
        "Class": ["Malicious"] * n_samples,
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Mock ransomware samples saved to: {output_path}")
    return df
