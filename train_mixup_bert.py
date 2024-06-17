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

# Mixup function
def mixup_data(x, y, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, 2)  # Binary classification
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def mixup_bert(train_df, val_df):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True)

    train_dataset = TextDataset(train_encodings, train_df['label'].tolist())
    val_dataset = TextDataset(val_encodings, val_df['label'].tolist())

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = BertClassifier(BertModel.from_pretrained("bert-base-cased"))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get embeddings and apply Mixup
            embeddings = model.get_embeddings(input_ids, attention_mask)
            embeddings, labels_a, labels_b, lam = mixup_data(embeddings, labels)
            
            outputs = model.classifier(embeddings)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}')

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_roc_auc = roc_auc_score(all_labels, all_preds)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation ROC AUC: {val_roc_auc}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model, tokenizer, device, criterion
