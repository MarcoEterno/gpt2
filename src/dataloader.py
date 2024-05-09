from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from torch import manual_seed

from src.config import BATCH_SIZE
from src.model.tokenizer import tokenize

# Load the 'train' split
original_train_ds = load_dataset("wikipedia", '20220301.en', split="train", trust_remote_code=True).with_format("torch")

# Set the seed for reproducibility and for not contaminating training data with validation data
manual_seed(42)

# Calculate the lengths of the splits
dataset_len = len(original_train_ds['id'])
train_len = int(dataset_len * 0.8)
validation_len = int(dataset_len * 0.1)
test_len = dataset_len - train_len - validation_len

# Create an 80/10/10 split for the training/validation/test sets
train_ds, validation_ds, test_ds = random_split(original_train_ds, [train_len, validation_len, test_len])
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tokenize)
validation_dataloader = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tokenize)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tokenize)

if __name__ == '__main__':
    print(original_train_ds)
    print(len(validation_ds))
    print(len(test_ds))
    print(train_dataloader.__dict__)
    print(next(enumerate(validation_dataloader)))
    print(len(test_dataloader))
