import logging
import os
import time
from datetime import datetime
from typing import Optional


import torch
from torch import optim, load, save
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import BATCH_SIZE
from src.config import CHECKPOINTS_DIR
from src.config import CONTEXT_LENGTH
from src.config import DEVICE
from src.config import EMBEDDING_SIZE
from src.config import NUM_DECODERS
from src.config import NUM_HEADS
from src.config import POSITIONAL_ENCODING_COEFFICIENT
from src.config import POSITIONAL_ENCODING_SCALAR
from src.config import SAVE_CHECKPOINT_EVERY_N_MINUTES
from src.config import TRAIN_NUM_EPOCHS
from src.config import VOCAB_SIZE
from src.model.gpt2 import GPT2
from src.utils.dataset import train_dataloader, validation_dataloader
from src.utils.tokenizer import tokenize, tokenizer


def lr_rate(step_num, d_model, factor, warmup_steps):
    step_num = max(1, step_num)
    return factor * (
            d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    )


def load_model(
        model: GPT2,
        start_epoch: int = 0,
        batch_number: int = 0,
        checkpoints_dir: Optional[str] = CHECKPOINTS_DIR
):
    """
    If resuming training, this function loads the state of the model and optimizer.

    Args:
        model (GPT2): The model into which the state is loaded.
        start_epoch (int, optional): If set to n, the model is loaded from the n-th epoch. Defaults to 0.
        batch_number (int, optional): If set to n, the model is loaded from the n-th batch. Defaults to 0.
        checkpoints_dir (str, optional): The directory where the checkpoints are stored. Defaults to CHECKPOINTS_DIR.

    Returns:
        GPT2: The initialized model.
        torch.optim.Adam: The optimizer with the state loaded.
        int: The epoch from which the model was initialized.
        int: The first batch of data that the model has not seen yet.
    """
    # initialize the optimizer to load if no checkpoint is found
    initial_optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    if not os.path.exists(checkpoints_dir):
        logging.log(logging.INFO,
                    f"Checkpoints directory not found, creating it, and loading the model randomly initialized")
        os.mkdir(checkpoints_dir)
        return model, initial_optimizer,0, 0
    if os.path.exists(checkpoints_dir):
        # find the last checkpoint that we saved and load it
        for epoch in range(start_epoch, 0, -1):
            for batch in range(start_batch_number, 0, -1):
                tentative_last_checkpoint_path = os.path.join(checkpoints_dir,
                                                              f"ckpt_"
                                                              f"{epoch}_"
                                                              f"{batch}_"
                                                              f"{CONTEXT_LENGTH}_"
                                                              f"{VOCAB_SIZE}_"
                                                              f"{EMBEDDING_SIZE}_"
                                                              f"{NUM_DECODERS}_"
                                                              f"{NUM_HEADS}.pt"
                                                              )
                if os.path.exists(tentative_last_checkpoint_path):
                    logging.log(logging.INFO, f"Loading the model from {tentative_last_checkpoint_path}")
                    checkpoint = load(tentative_last_checkpoint_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer = model.load_state_dict(checkpoint['optimizer_state_dict'])

                    return model, optimizer, start_epoch, batch + 1 # in principle it is possible that the next batch number
                    # does not exist, and the model was saved right at the end of a training epoch. but this is unlikely

        # if no checkpoints are found, load the model randomly initialized
        logging.log(logging.INFO, f"No checkpoints found, loading the model randomly initialized")
        return model, initial_optimizer, 0, 0


def save_model(
        model: GPT2,
        epoch: int,
        batch_number: int,
        loss: float, checkpoints_dir:
        Optional[str] = CHECKPOINTS_DIR
):
    """
    Save the model to the checkpoints directory.
    
    Args:
        model (GPT2): The model to be saved.
        epoch (int): The epoch at which the model is saved.
        batch_number (int): The batch number at which the model is saved.
        loss (float): The loss at which the model is saved.
        checkpoints_dir (Optional[str], optional): The directory where the model is saved. Defaults to CHECKPOINTS_DIR.
    
    Returns:
        None
    """
    # this function can be made asynchronous in case the model is too large to save in a reasonable time
    if not os.path.exists(checkpoints_dir):
        logging.log(logging.INFO, f"Checkpoints directory not found, creating it, and saving the model")
        os.mkdir(checkpoints_dir)
    if os.path.exists(checkpoints_dir):
        checkpoint_path = os.path.join(checkpoints_dir,
                                       f"ckpt_"
                                       f"{epoch}_"
                                       f"{batch}_"
                                       f"{CONTEXT_LENGTH}_"
                                       f"{VOCAB_SIZE}_"
                                       f"{EMBEDDING_SIZE}_"
                                       f"{NUM_DECODERS}_"
                                       f"{NUM_HEADS}.pt"
                                       )
        logging.log(logging.INFO, f"Saved checkpoint to {checkpoint_path}")
        save({
            'epoch': epoch,
            'batch_number': batch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)


if __name__ == "__main__":
    start_epoch = 0
    start_batch_number = 0
    model_parameters = f"_{BATCH_SIZE}_{CONTEXT_LENGTH}_{VOCAB_SIZE}_{EMBEDDING_SIZE}_{NUM_DECODERS}_{NUM_HEADS}"

    logdir = "logs/logs" + datetime.now().strftime("%d%m-%H:%M:%S") + str(model_parameters) + "/"
    writer = SummaryWriter(log_dir=logdir)
    logging.basicConfig(level=logging.INFO)
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

    if start_epoch > 0 or start_batch_number > 0:
        model, optimizer, start_epoch, start_batch_number = load_model(model, start_epoch, start_batch_number)
    else:
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    time_of_last_checkpoint_ns = time.perf_counter_ns()
    logging.log(logging.INFO, f"Starting training")

    for epoch in range(start_epoch, TRAIN_NUM_EPOCHS):  # loop over the dataset multiple times
        for i, batch in enumerate(train_dataloader, start=start_batch_number):
            inputs = batch
            labels = batch['input_ids']
            running_training_loss = 0.0

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
                if time_from_last_checkpoint_ns > 60 * 1e9 * SAVE_CHECKPOINT_EVERY_N_MINUTES:
                    save_model(model, epoch, i, loss.item())
                    time_of_last_checkpoint_ns = time.perf_counter_ns()
                    logging.log(logging.INFO, f"Saved checkpoint at epoch {epoch} and step {i}")

                # evaluate the model on the validation set
                model.eval()
                with torch.no_grad():
                    validation_batch = next(iter(validation_dataloader))
                    validation_inputs = validation_batch
                    validation_labels = validation_batch['input_ids']
                    validation_outputs = model(validation_inputs, j)
                    one_hot_validation_labels = F.one_hot(validation_labels[:, j], num_classes=VOCAB_SIZE).float()
                    validation_loss = F.cross_entropy(validation_outputs, one_hot_validation_labels)

                model.train()
                # print the output of the model
                if True:
                    probabilities = F.softmax(outputs, dim=1)
                    logging.log(
                        logging.INFO,
                        f'Sentence:{tokenizer.decode(batch["input_ids"][0][:j])}\n'
                        f'Prediction: {tokenizer.decode(outputs[0].argmax().item())}\n'
                        f'Probability: {probabilities[0].max().item()}\n'  # evaluate whether to print top k guesses
                                )


                # log training statistics to tensorboard
                # writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], i)
                writer.add_scalar('Loss/train', loss.item(), i)
                writer.add_scalar('Loss/validation', validation_loss.item(), i)

                running_training_loss += loss.item()
            if True:  # print every 5 mini-batches
                logging.log(logging.INFO, f'[{epoch + 1}, {i + 1:5d}] loss: {running_training_loss / 20:.3f}')

    logging.log(logging.INFO, 'Finished Training')
    save_model(model, epoch, i, loss.item())
    writer.close()
