from datasets import load_dataset

train_ds = load_dataset("wikipedia", '20220301.en', split="train", trust_remote_code=True).with_format("torch")


if __name__ == '__main__':
    print(train_ds)
    print(train_ds[0])
