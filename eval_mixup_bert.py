import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_mixup_bert(model, test_data, tokenizer, device, criterion):
    test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True)
    test_dataset = TextDataset(test_encodings, test_data['label'].tolist())
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    avg_test_loss = test_loss / len(test_dataloader)
    test_roc_auc = roc_auc_score(all_labels, all_preds)
    print(f'Test Loss: {avg_test_loss}, Test ROC AUC: {test_roc_auc}')
    return avg_test_loss, test_roc_auc