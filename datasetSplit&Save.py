import pandas as pd
from datasets import load_dataset

# Load and preprocess the dataset
dataset = load_dataset("pirocheto/phishing-url")

def process_dataset(split):
    df = dataset[split].to_pandas()
    df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})
    df = df.rename(columns={'status': 'label'})
    df.to_csv(f'phishing_url_{split}.csv', index=False)
    return df


train_df = process_dataset('train')
test_df = process_dataset('test')