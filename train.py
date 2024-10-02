import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F
from IIR_Preprocessing.MyDataset import MyDataset
from IIR_Preprocessing.BertSentimentClassifier import BertSentimentClassifier
from IIR_Preprocessing.utils import *
import warnings
import wandb
warnings.filterwarnings("ignore")

# Initialize wandb
path_data_train = "IIR_Preprocessing/save_results/train_aug_llamba.csv"
project_name = os.path.basename(path_data_train).replace('.csv', '')
name_project = "Thesis_SemEval"
wandb.init(
    project="Thesis_SemEval",
    name=project_name, 
    config={
        "learning_rate": 2e-5,
        "epochs": 150,
        "batch_size": 400
})

name_bert = "alvaroalon2/biobert_diseases_ner"
tokenizer = BertTokenizer.from_pretrained(name_bert) 

# Load data
path_data_test = "IIR_Preprocessing/save_results/test_dataset.csv"
path_data_devel = "IIR_Preprocessing/save_results/Devel_dataset.csv"
train_df = pd.read_csv(path_data_train, sep=',', encoding='utf-8')
test_df = pd.read_csv(path_data_test, sep=',', encoding='utf-8')
devel_df = pd.read_csv(path_data_devel, sep=',', encoding='utf-8')

# Label encoding
le = LabelEncoder()
class_mapping = {
    'false': 0,
    'effect': 1,
    'mechanism': 2,
    'advise': 3,
    'int': 4
}
train_df['pair type'] = le.fit_transform(train_df['pair type'].map(class_mapping))
test_df['pair type'] = le.fit_transform(test_df['pair type'].map(class_mapping))
devel_df['pair type'] = le.fit_transform(devel_df['pair type'].map(class_mapping))

# Split data
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['pair type'], random_state=42)

# Create data loaders
max_len = 50
batch_size = 400
num_epochs = 150  # Adjust as needed

train_dataset = MyDataset(train_df, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

devel_dataset = MyDataset(devel_df, tokenizer, max_len)
devel_loader = DataLoader(devel_dataset, batch_size=batch_size)

test_dataset = MyDataset(test_df, tokenizer, max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

val_dataset = MyDataset(val_df, tokenizer, max_len)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define model and optimizer
num_classes = train_df["pair type"].nunique()
model = BertSentimentClassifier(name_bert, num_classes)
criterion = nn.CrossEntropyLoss()

# Create logs folder
logs_folder = "logs"
os.makedirs(logs_folder, exist_ok=True)

# Generate a filename for logging
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{logs_folder}/training_log_BERT1_{current_time}.txt"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 2e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.to(device)

# Metrics lists
train_losses, train_precisions, train_recalls, train_f1_scores = [], [], [], []
devel_losses, devel_precisions, devel_recalls, devel_f1_scores = [], [], [], []
test_losses, test_precisions, test_recalls, test_f1_scores = [], [], [], []
val_losses, val_precisions, val_recalls, val_f1_scores = [], [], [], []

best_test_loss = float('inf')
best_epoch = 0

# Start training loop
with open(log_filename, 'a') as log_file:
    for epoch in range(num_epochs):
        start_time = datetime.now()
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Training
        train_loss, train_precision, train_recall, train_f1 = train(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)

        # Development
        devel_loss, devel_precision, devel_recall, devel_f1 = evaluate(model, devel_loader, criterion, device)

        # Testing
        test_loss, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)

        # Log results
        wandb.log({
            'Train Loss': train_loss, 'Train Precision': train_precision, 'Train Recall': train_recall, 'Train F1': train_f1,
            'Val Loss': val_loss, 'Val Precision': val_precision, 'Val Recall': val_recall, 'Val F1': val_f1,
            'Devel Loss': devel_loss, 'Devel Precision': devel_precision, 'Devel Recall': devel_recall, 'Devel F1': devel_f1,
            'Test Loss': test_loss, 'Test Precision': test_precision, 'Test Recall': test_recall, 'Test F1': test_f1
        })

        print(f'Train -- Loss: {train_loss:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}')
        print(f'Val   -- Loss: {val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}')
        print(f'Devel -- Loss: {devel_loss:.4f} | Precision: {devel_precision:.4f} | Recall: {devel_recall:.4f} | F1: {devel_f1:.4f}')
        print(f'Test  -- Loss: {test_loss:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}')

        log_file.write(f'Epoch {epoch + 1}/{num_epochs}\n')
        log_file.write(f'Train -- Loss: {train_loss:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}\n')
        log_file.write(f'Val   -- Loss: {val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}')
        log_file.write(f'Devel -- Loss: {devel_loss:.4f} | Precision: {devel_precision:.4f} | Recall: {devel_recall:.4f} | F1: {devel_f1:.4f}\n')
        log_file.write(f'Test  -- Loss: {test_loss:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}\n')
        log_file.flush()

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'best_model_BERT1.pt')

        # Save metrics
        train_losses.append(train_loss)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1_scores.append(train_f1)

        val_losses.append(val_loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)

        devel_losses.append(devel_loss)
        devel_precisions.append(devel_precision)
        devel_recalls.append(devel_recall)
        devel_f1_scores.append(devel_f1)

        test_losses.append(test_loss)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1)

        # Calculate and print time remaining
        elapsed_time = datetime.now() - start_time
        remaining_time = (num_epochs - epoch - 1) * elapsed_time
        print(f'Time Elapsed: {elapsed_time}, Time Remaining: {remaining_time} \n')
        torch.save(model.state_dict(), f'final_model_BERT1.pt')

# Print the best epoch
print(f"Best Model: Epoch {best_epoch}, Best Validation Loss: {best_test_loss:.4f}")
print(f"Path log file: {log_filename}")
