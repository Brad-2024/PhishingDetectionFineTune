from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, AdamW, DistilBertModel
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, Repository
from huggingface_hub import login
from torch.utils.tensorboard import SummaryWriter

# Login to the Hugging Face Hub
login('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')
writer = SummaryWriter(log_dir='./new_runs')



# Load the dataset
dataset = load_dataset("pirocheto/phishing-url")


# Process the dataset
def process_dataset(split):
    df = dataset[split].to_pandas()
    df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})
    df = df.rename(columns={'status': 'label'})
    df.to_csv(f'phishing_url_{split}.csv', index=False)
    return df


train_df = process_dataset('train')
test_df = process_dataset('test')

