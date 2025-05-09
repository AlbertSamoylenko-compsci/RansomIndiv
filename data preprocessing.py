import pandas as pd

def preprocess_malmem_data(file_path: str, correlation_threshold: float = 0.85):
    #load csv
    df = pd.read_csv(file_path)
    print("Original shape:", df.shape)

    #drop non-informative id-like columns
    df = df.drop(columns=["Category", "Filename"])

    #drop numeric columns where all values are zero
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    zero_cols = [col for col in numeric_cols if (df[col] == 0).all()]
    print("Dropping all-zero columns:", zero_cols)
    df = df.drop(columns=zero_cols)
    print("Shape after dropping all-zero columns:", df.shape)

    #encode 'class' as binary: benign = 0, malicious = 1
    df["Class"] = df["Class"].apply(lambda x: 0 if x == "Benign" else 1)

    #drop features too strongly correlated with 'class'
    corrs = df.corr(numeric_only=True)["Class"].abs()
    leaky_cols = corrs[corrs > correlation_threshold].index.tolist()
    if "Class" in leaky_cols:
        leaky_cols.remove("Class")

    print(f"Dropping highly correlated features (>{correlation_threshold}):", leaky_cols)
    df = df.drop(columns=leaky_cols)

    #split into features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    print("Final shape of features:", X.shape)
    return X, y

X, y = preprocess_malmem_data("MalMem2022.csv", correlation_threshold=0.85)

