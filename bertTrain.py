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

# Tokenize URLs and scale features
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokenize_and_scale_features(df):
    tokenized_data = tokenizer(df['url'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    non_textual_features = df.drop(columns=['url', 'label'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(non_textual_features)
    labels = torch.tensor(df['label'].values, dtype=torch.float32)
    return tokenized_data, torch.tensor(scaled_features, dtype=torch.float32), labels


tokenized_train_data, scaled_train_features, train_labels = tokenize_and_scale_features(train_df)
tokenized_test_data, scaled_test_features, test_labels = tokenize_and_scale_features(test_df)

# Create TensorDataset and DataLoader for training and testing
train_dataset = TensorDataset(
    tokenized_train_data['input_ids'],
    tokenized_train_data['attention_mask'],
    scaled_train_features,
    train_labels
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(
    tokenized_test_data['input_ids'],
    tokenized_test_data['attention_mask'],
    scaled_test_features,
    test_labels
)
test_loader = DataLoader(test_dataset, batch_size=32)


# Define the custom model
class PhishingDetectionModel(nn.Module):
    def __init__(self, num_non_text_features):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 + num_non_text_features, 1)

    def forward(self, input_ids, attention_mask, non_text_features):
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = distilbert_output.last_hidden_state[:, 0, :]
        combined_features = torch.cat((self.dropout(pooled_output), non_text_features), dim=1)
        logits = self.classifier(combined_features)
        return logits


# Initialize model, optimizer, and loss function
model = PhishingDetectionModel(num_non_text_features=scaled_train_features.shape[1])
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    total_eval_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, non_text_features, labels = [b.to(device) for b in batch]

            outputs = model(input_ids, attention_mask, non_text_features)
            loss = loss_fn(outputs.squeeze(), labels)
            total_eval_loss += loss.item()

            preds = torch.round(torch.sigmoid(outputs.squeeze()))
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    avg_eval_loss = total_eval_loss / len(data_loader)

    return accuracy, precision, recall, f1, avg_eval_loss


# Training and Evaluation Loop
num_epochs = 3
for epoch in range(num_epochs):
    # Training step
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        input_ids, attention_mask, non_text_features, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, non_text_features)
        loss = loss_fn(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss}")

    # Evaluation step
    accuracy, precision, recall, f1, avg_eval_loss = evaluate(model, test_loader, device)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_eval_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
