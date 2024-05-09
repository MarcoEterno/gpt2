import os
import logging
from typing import Optional

import torch
from torch import optim, load, save

from src.config import CHECKPOINTS_DIR
from src.config import CONTEXT_LENGTH
from src.config import VOCAB_SIZE
from src.config import EMBEDDING_SIZE
from src.config import NUM_DECODERS
from src.config import NUM_HEADS
from src.model.gpt2 import GPT2


def load_model(model, start_epoch: int = 0, start_batch_number: int = 0, checkpoints_dir: str = CHECKPOINTS_DIR):
    """
    Load the model and optimizer state from a checkpoint.

    Args:
        model (torch.nn.Module): The model instance.
        start_epoch (int): Starting epoch number for loading.
        start_batch_number (int): Starting batch number for loading.
        checkpoints_dir (str): Directory containing the checkpoints.

    Returns:
        tuple: Loaded model, optimizer, starting epoch, and batch number.
    """
    # Check if checkpoints directory exists
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        logging.info("Checkpoints directory created at {}".format(checkpoints_dir))
        optimizer = optim.Adam(model.parameters())
        return model, optimizer, 0, 0

    # Attempt to find the latest checkpoint
    checkpoint_path = find_latest_checkpoint(checkpoints_dir, start_epoch, start_batch_number)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters())  # Initialize optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return model, optimizer, checkpoint['epoch'], checkpoint['batch_number']
    else:
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
        return model, optimizer, 0, 0


def save_model(model, optimizer, epoch, batch_number, loss, checkpoints_dir=CHECKPOINTS_DIR):
    """
    Save the model and optimizer state to a checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        batch_number (int): Current batch number.
        loss (float): Current loss.
        checkpoints_dir (str): Directory to save the checkpoints.
    """
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        logging.info("Checkpoints directory created at {}".format(checkpoints_dir))

    checkpoint_path = os.path.join(checkpoints_dir,
                                   f"ckpt_"
                                   f"{epoch}_"
                                   f"{batch_number}_"
                                   f"{CONTEXT_LENGTH}_"
                                   f"{VOCAB_SIZE}_"
                                   f"{EMBEDDING_SIZE}_"
                                   f"{NUM_DECODERS}_"
                                   f"{NUM_HEADS}.pt"
                                   )
    torch.save({
        'epoch': epoch,
        'batch_number': batch_number,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logging.info("Saved checkpoint at '{}'".format(checkpoint_path))


def find_latest_checkpoint(start_epoch: int, start_batch_number: int, checkpoints_dir: str = CHECKPOINTS_DIR):
    """
    Find the latest checkpoint in the directory starting from a specific epoch and batch number.

    Args:
        start_epoch (int): Starting epoch number to look for checkpoints.
        start_batch_number (int): Starting batch number to look for checkpoints.
        checkpoints_dir (str): The directory where checkpoints are stored.

    Returns:
        str: The path to the latest found checkpoint or None if no checkpoint is found.
    """
    for epoch in range(start_epoch, 0, -1):
        for batch in range(start_batch_number, 0, -1):
            checkpoint_path = os.path.join(
                checkpoints_dir,
                f"ckpt_"
                f"{epoch}_"
                f"{batch}_"
                f"{CONTEXT_LENGTH}_"
                f"{VOCAB_SIZE}_"
                f"{EMBEDDING_SIZE}_"
                f"{NUM_DECODERS}_"
                f"{NUM_HEADS}.pt"
            )
            if os.path.exists(checkpoint_path):
                return checkpoint_path
    return None


def load_model_old(
        model: GPT2,
        start_epoch: int = 0,
        start_batch_number: int = 0,
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
        return model, initial_optimizer, 0, 0
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

                    return model, optimizer, start_epoch, batch + 1  # in principle it is possible that the next batch number
                    # does not exist, and the model was saved right at the end of a training epoch. but this is unlikely

        # if no checkpoints are found, load the model randomly initialized
        logging.log(logging.INFO, f"No checkpoints found, loading the model randomly initialized")
        return model, initial_optimizer, 0, 0


def save_model_old(
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
                                       f"{batch_number}_"
                                       f"{CONTEXT_LENGTH}_"
                                       f"{VOCAB_SIZE}_"
                                       f"{EMBEDDING_SIZE}_"
                                       f"{NUM_DECODERS}_"
                                       f"{NUM_HEADS}.pt"
                                       )
        logging.log(logging.INFO, f"Saved checkpoint to {checkpoint_path}")
        save({
            'epoch': epoch,
            'batch_number': batch_number,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)