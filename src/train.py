import time
import logging
from datetime import datetime

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import BATCH_SIZE
from src.config import SAVE_CHECKPOINT_EVERY_N_MINUTES
from src.config import CONTEXT_LENGTH
from src.config import DEVICE
from src.config import EMBEDDING_SIZE
from src.config import NUM_DECODERS
from src.config import NUM_HEADS
from src.config import POSITIONAL_ENCODING_COEFFICIENT
from src.config import POSITIONAL_ENCODING_SCALAR
from src.config import TRAIN_NUM_EPOCHS
from src.config import VOCAB_SIZE
from src.model.gpt2 import GPT2
from src.utils.dataset import train_ds
from src.utils.tokenizer import tokenize, tokenizer




def lr_rate(step_num, d_model, factor, warmup_steps):
    step_num = max(1, step_num)
    return factor * (
            d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    )


def save_checkpoint(model, optimizer, epoch, batch_number, loss, path=None):
    if path is None:
        path = f"checkpoints/ckpt_{epoch}_{batch_number}_{CONTEXT_LENGTH}_{VOCAB_SIZE}_{EMBEDDING_SIZE}_{NUM_DECODERS}_{NUM_HEADS}.pt"  # additional infos like timestamp and model features can be added
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logging.log(logging.INFO, f"Saved checkpoint to {path}")


def load_checkpoint(model, epoch, batch_number,  path=None):
    if path is None:
        path = f"checkpoints/ckpt_{epoch}_{batch_number}_{CONTEXT_LENGTH}_{VOCAB_SIZE}_{EMBEDDING_SIZE}_{NUM_DECODERS}_{NUM_HEADS}.pt"  # additional infos like timestamp and model features can be added
    checkpoint = torch.load(path)  # add map_location=DEVICE if needed to speed up loading
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.log(logging.INFO, f"Loaded checkpoint from {path}")
    return model, optimizer, lr_scheduler, epoch, loss

if __name__ == "__main__":
    start_epoch = 0
    start_batch_number = 0

    logdir = "logs/logs" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = SummaryWriter(log_dir=logdir)
    logging.basicConfig(level=logging.INFO)

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tokenize)
    model = GPT2(
        vocabulary_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        context_length=CONTEXT_LENGTH,
        positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
        positional_encoding_coefficient=POSITIONAL_ENCODING_COEFFICIENT,
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        num_decoders=NUM_DECODERS
    ).to(DEVICE)

    if start_epoch > 0:
        model, optimizer, lr_scheduler, epoch, loss = load_checkpoint(model, start_epoch, start_batch_number)
    else:
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
        #lr_scheduler = optim.lr_scheduler.ConstantLR(
        #    optimizer=optimizer,
        #    factor=0.001,
        #    total_iters=1000
        #    )
        #lr_scheduler = optim.lr_scheduler.LambdaLR(
        #    optimizer=optimizer,
        #    lr_lambda=lambda step_num: lr_rate(
        #        step_num, d_model=512, factor=1, warmup_steps=4000
        #    ),
        #)
    time_of_last_checkpoint_ns = time.perf_counter_ns()
    print("Starting training")
    # attention: this training does not take into account that the model can be loaded from memory.
    # if it is, the model sees the same tokens multiple times, overfitting the same data. probably shiffle=true should be used while dataloading
    for epoch in range(TRAIN_NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            inputs = batch
            labels = batch['input_ids']

            # loop over all the words in the current batch
            for j in range(CONTEXT_LENGTH):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs, j)
                one_hot_labels = F.one_hot(labels[:, j], num_classes=VOCAB_SIZE).float()
                loss = F.cross_entropy(outputs, one_hot_labels)
                loss.backward()
                optimizer.step()

                # save a model checkpoint every N minutes
                time_from_last_checkpoint_ns = time.perf_counter_ns() - time_of_last_checkpoint_ns
                if time_from_last_checkpoint_ns > 60 * 1e9 * SAVE_CHECKPOINT_EVERY_N_MINUTES :
                    save_checkpoint(model, optimizer, epoch, i, loss)
                    time_of_last_checkpoint_ns = time.perf_counter_ns()
                    print(f"Saved checkpoint at epoch {epoch} and step {i}")

                # print the output of the model
                if i % 20 == 19:
                    print(
                        f'Sentence:{tokenizer.decode(batch["input_ids"][0][:j])}, \n'
                        f'Prediction: {tokenizer.decode(outputs[0].argmax().item())}'
                    )
                    print(outputs)
                    print()

                # log training statistics to tensorboard
                writer.add_scalar('Loss/train', loss.item(), i)
                writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], i)
                writer.add_scalar('Epoch', epoch, i)


                running_loss += loss.item()
            if i % 20 == 19:  # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                # print(inputs['input_ids'].shape, labels.shape) # torch.Size([BATCH_SIZE, CONTEXT LENGTH])

                # print(tokenizer.decode(inputs[i][0], labels[i][0]))
                running_loss = 0.0

    logging.info('Finished Training')
    save_checkpoint(model, optimizer, lr_scheduler, epoch, j, loss)
    writer.close()
