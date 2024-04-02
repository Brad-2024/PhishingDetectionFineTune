# Imports
import os
from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, Repository, login, create_repo, upload_file
from torch.utils.tensorboard import SummaryWriter

# Login to the Hugging Face Hub
login('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')
writer = SummaryWriter(log_dir='./new_runs')

train_df = pd.read_csv('phishing_url_train.csv')
test_df = pd.read_csv('phishing_url_test.csv')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


# Define function to tokenize URLs
def tokenize_urls(urls):
    return tokenizer(urls, padding=True, truncation=True, return_tensors='pt')


# Extract numerical features and scale them
def extract_and_scale_features(df):
    numerical_features = df.drop(columns=['url', 'label'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numerical_features)
    return torch.tensor(scaled_features, dtype=torch.float)


# Tokenize URLs and extract numerical features
train_tokens = tokenize_urls(train_df['url'].tolist())
test_tokens = tokenize_urls(test_df['url'].tolist())
train_numerical_features = extract_and_scale_features(train_df)
test_numerical_features = extract_and_scale_features(test_df)


# Custom Dataset Class
class PhishingURLDataset(Dataset):
    def __init__(self, tokens, numerical_features, labels):
        self.tokens = tokens
        self.numerical_features = numerical_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokens.items()}
        item['numerical_features'] = self.numerical_features[idx]
        item['labels'] = self.labels[idx].detach().clone()
        return item


# Prepare the datasets and dataloaders
train_labels = torch.tensor(train_df['label'].values, dtype=torch.long)
test_labels = torch.tensor(test_df['label'].values, dtype=torch.long)
train_dataset = PhishingURLDataset(train_tokens, train_numerical_features, train_labels)
test_dataset = PhishingURLDataset(test_tokens, test_numerical_features, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Custom Model
class BERTLargeWithFeatures(nn.Module):
    def __init__(self, num_numerical_features):
        super(BERTLargeWithFeatures, self).__init__()
        self.bert = BertModel.from_pretrained("bert-large-uncased")
        self.num_features_processor = nn.Linear(num_numerical_features, 128)
        self.classifier = nn.Linear(1152, 2)

    def forward(self, input_ids, attention_mask, numerical_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        numerical_features = self.num_features_processor(numerical_features)
        combined_features = torch.cat((pooled_output, numerical_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, numerical_features)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    return accuracy, precision, recall, f1

# Training
model = BERTLargeWithFeatures(num_numerical_features=train_numerical_features.shape[1])
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask, numerical_features)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    # Calculate training loss for the epoch
    avg_loss = total_loss / len(train_loader)

    # Evaluate the model on the test set after each epoch
    accuracy, precision, recall, f1 = evaluate(model, test_loader, device)

    # Print metrics
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Training Loss: {avg_loss:.4f}")
    print(
        f"Validation Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")



# Evaluate
def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, numerical_features)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, f1


accuracy, precision, recall, f1 = evaluate(model, test_loader, device)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Save and Upload
model_path = "bertLarge_phish"
tokenizer.save_pretrained(model_path)
torch.save(model.state_dict(), f"{model_path}/pytorch_model.bin")