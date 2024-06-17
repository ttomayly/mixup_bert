import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, 2)  # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def evaluate_bert(model, test_df, tokenizer, device, criterion, batch_size=16, max_length=128):
    test_dataset = TextDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    test_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating Test Set'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy()[:, 1])  # assuming binary classification, take the second column for positive class

    avg_test_loss = test_loss / len(test_dataloader)

    test_roc_auc = roc_auc_score(all_labels, all_predictions)
    print(f'Test Loss: {avg_test_loss}, Test ROC AUC: {test_roc_auc}')

    return avg_test_loss, test_roc_auc