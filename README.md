# BERT Classification on Rotten Tomatoes Dataset

## Description

This repository contains the code and instructions for training a BERT-based sentiment classification model on the [Rotten Tomatoes Dataset](https://huggingface.co/datasets/rotten_tomatoes).

The training pipeline includes bert model and model evaluation on a held-out validation set. 
 
In addition, though it get us worse reults, the code includes data augmentation using a custom MixUp method at the embedding level and also model evaluation on a held-out validation set.

The best-performing model is saved as `best_model.pt`.

## Results

### Without Augmentation on Train/Validation Set after Training
- **Train Loss:** 0.4
- **Val Loss:** 0.45
- **Val ROC-AUC:** 0.88

### Without Augmentation on Test Set
- **Test Loss:** 0.46
- **Test ROC-AUC:** 0.88

### With Augmentation on Train/Validation Set after Training
- **Train Loss:** 0.66
- **Val Loss:** 0.65
- **Val ROC-AUC:** 0.66

### With Augmentation on Test Set
- **Test Loss:** 0.63
- **Test ROC-AUC:** 0.70


## Discussion

The performance of the model with MixUp data augmentation was worse compared to training without augmentation. This decline can be attributed to several factors:

1. **Loss of Context**: MixUp creates samples that blend meanings of different words or sentences, leading to loss of crucial context and interpretability.
2. **Model Robustness**: a high training loss indicates that model maybe because of noise becuase of augmentation or becuase of underfitting and thus the model paramaters should be tuned.

Preprocessing was also minimized as more aggressive preprocessing led to worse performance, likely due to the removal of important contextual words.

## Conclusion
Future work could explore alternative augmentation techniques and more sophisticated preprocessing methods to enhance model performance. Other augmentation techniques such as back-translation, synonym replacement, or contextual word substitution might yield better results.

---

**Repository Structure:**

- `train_bert.py`: Script for training the BERT model without augmentation.
- `train_mixup_bert.py`: Script for training the BERT model with MixUp augmentation.
- `eval_bert.py`: Script for evaluating the standard BERT model.
- `eval_mixup_bert.py`: Script for evaluating the BERT model trained with MixUp augmentation.
- `preprocessing.py`: Utility functions for data preprocessing.
- `samokat EDA.ipynb`: Jupyter notebook for exploratory data analysis.
- `best_model.pt`: The best-performing model saved during training.
- `dataset/`: Directory containing the dataset.
