import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os

# Define the custom model class (Ensure this matches the class used during training)
class DistilBERTWithFeatures(torch.nn.Module):
    def __init__(self, num_numerical_features):
        super(DistilBERTWithFeatures, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=False)
        self.num_features_processor = torch.nn.Linear(num_numerical_features, 128)
        self.classifier = torch.nn.Linear(768 + 128, 2)

    def forward(self, input_ids, attention_mask, numerical_features):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = distilbert_output[0][:, 0, :]
        numerical_features = self.num_features_processor(numerical_features)
        combined_features = torch.cat((pooled_output, numerical_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

# Assuming the DistilBertTokenizer and DistilBertModel are already imported
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Function to preprocess data
def preprocess_data(df, tokenizer, scaler=None):
    tokens = tokenizer(df['url'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
    numerical_features = df.drop(columns=['url', 'label']).values
    if scaler is None:
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(numerical_features)
    else:
        numerical_features = scaler.transform(numerical_features)
    numerical_features = torch.tensor(numerical_features, dtype=torch.float)
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    return tokens, numerical_features, labels

# Custom Dataset for DataLoader
class CustomDataset(Dataset):
    def __init__(self, tokens, numerical_features, labels):
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.numerical_features = numerical_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "numerical_features": self.numerical_features[idx],
            "labels": self.labels[idx]
        }

# Load the test dataset
test_df = pd.read_csv('phishing_url_test.csv')  # Update this path

# Preprocess the test dataset
test_tokens, test_numerical_features, test_labels = preprocess_data(test_df, tokenizer)

# Prepare the DataLoader
test_dataset = CustomDataset(test_tokens, test_numerical_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the trained model
print("Current working directory:", os.getcwd())
model_path = '/root/PycharmProjects/dBertPhish/distilbert_phish'  # Update this path
num_numerical_features = test_numerical_features.shape[1]
model = DistilBERTWithFeatures(num_numerical_features=num_numerical_features)
model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
model.eval()

# Evaluation
correct_predictions = 0
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch['input_ids'], batch['attention_mask'], batch['numerical_features'])
        _, predictions = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(predictions == batch['labels']).item()

accuracy = correct_predictions / len(test_dataset)
print(f'Accuracy: {accuracy:.4f}')