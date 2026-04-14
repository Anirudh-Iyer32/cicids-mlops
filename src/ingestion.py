import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Data loaded successfully")
    print(df.shape)
    return df

if __name__ == "__main__":
    df = load_data("data/raw/wednesday.csv")
    print(df.head())