import os
from huggingface_hub import HfApi, Repository, login, create_repo, upload_file

# Login to the Hugging Face Hub
login('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')

# Your Hugging Face username
username = "Brad-2024"
# The name of your model, this will also be the repository name
model_name = "distilbert_phish"

# Initialize the HfApi instance
api = HfApi()

# Create a repository on Hugging Face Hub (if it doesn't already exist)
# This step is optional if you've already created the repository through the web UI
repo_url = create_repo(token=api.token, repo_id=model_name, exist_ok=True, private=True)

# Define the path to your local model and tokenizer files
model_file_path = "distilbert_phish/pytorch_model.bin"
config_file_path = "distilbert_phish/config.json"
tokenizer_files_path = ["distilbert_phish/tokenizer_config.json", "distilbert_phish/vocab.txt", "distilbert_phish/special_tokens_map.json"] # Add all tokenizer related files

# Upload the model file
upload_file(
    path_or_fileobj=model_file_path,
    path_in_repo="pytorch_model.bin",  # Path in the repository
    repo_id=f"{username}/{model_name}",
    token=api.token
)

# Upload the config file
upload_file(
    path_or_fileobj=config_file_path,
    path_in_repo="config.json",  # Path in the repository
    repo_id=f"{username}/{model_name}",
    token=api.token
)

# Upload tokenizer files
for file_path in tokenizer_files_path:
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),  # Use the file name as the path in the repository
        repo_id=f"{username}/{model_name}",
        token=api.token
    )