from preprocessing import preprocess

from train_mixup_bert import mixup_bert
from train_bert import bert

from eval_mixup_bert import evaluate_mixup_bert
from eval_bert import evaluate_bert

import pandas as pd

train_df = pd.read_parquet("dataset/train.parquet")
val_df = pd.read_parquet("dataset/validation.parquet")
test_df = pd.read_parquet("dataset/test.parquet")

pre_train_df = preprocess(train_df)
pre_val_df = preprocess(val_df)
pre_test_df = preprocess(test_df)

# Without Augmentation
model, tokenizer, device, criterion = bert(pre_train_df, pre_val_df)
avg_test_loss, test_roc_auc = evaluate_bert(model, pre_test_df, tokenizer, device, criterion)

# With Augmentation
model, tokenizer, device, criterion = mixup_bert(pre_train_df, pre_val_df)
avg_test_loss, test_roc_auc = evaluate_mixup_bert(model, pre_test_df, tokenizer, device, criterion)