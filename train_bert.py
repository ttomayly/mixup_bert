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

def bert(train_df, val_df, batch_size=16, max_length=128, num_epochs=2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertClassifier(BertModel.from_pretrained("bert-base-cased"))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    train_dataset = TextDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, max_length)
    val_dataset = TextDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    train_loss = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}'):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss/ len(train_dataloader)

        model.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy()[:, 1])  # assuming binary classification, take the second column for positive class

        avg_val_loss = val_loss / len(val_dataloader)

        roc_auc = roc_auc_score(all_labels, all_predictions)

        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, ROC AUC: {roc_auc}')

    return model, tokenizer, device, criterion
