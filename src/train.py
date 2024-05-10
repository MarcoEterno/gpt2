import logging
import time
from datetime import datetime

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter

from src.config import BATCH_SIZE
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
from src.dataloader import train_dataloader, validation_dataloader
from src.model.tokenizer import tokenizer
from src.checkpoint_management import load_model, save_model, load_model_old, save_model_old
# TODO: check generation probabilities after softmax with debugger, check if logging is done right,
# TODO: study batch size influence,

def lr_rate(step_num, d_model, factor, warmup_steps):
    #step_num = max(1, step_num)
    return 1 # factor * (d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

if __name__ == "__main__":
    start_epoch = 0
    start_batch_number = 109
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

    total_model_parameters = sum(p.numel() for p in model.parameters())

    if start_epoch > 0 or start_batch_number > 0:
        model, optimizer, start_epoch, start_batch_number = load_model_old(model, start_epoch, start_batch_number)
    else:
        optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.95), eps=1e-9, weight_decay=0.001)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_rate(step, total_model_parameters, 10e-2, 10))

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
                loss = F.cross_entropy(outputs, one_hot_labels) # TODO: if this is wrong it is enough to break the program, so double check
                loss.backward()
                optimizer.step()
                scheduler.step()

                # save a model checkpoint every N minutes
                time_from_last_checkpoint_ns = time.perf_counter_ns() - time_of_last_checkpoint_ns
                if time_from_last_checkpoint_ns > 60 * 1e9 * SAVE_CHECKPOINT_EVERY_N_MINUTES:
                    save_model(model, optimizer, epoch, i, loss.item())
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
                    #print(outputs.shape)
                    #print(outputs)
                    #print(probabilities[0])
                    #print(sum(probabilities[0]))
                    logging.log(
                        logging.INFO,
                        f'Sentence:{tokenizer.decode(batch["input_ids"][0][:j])}\n'
                        f'Prediction: {tokenizer.decode(outputs[0].argmax().item())}\n'
                        f'Probability: {probabilities[0].max().item()}\n'  # evaluate whether to print top k guesses
                    )


                # log training statistics to tensorboard
                writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], i)
                writer.add_scalar('Loss/train', loss.item(), i)
                writer.add_scalar('Loss/validation', validation_loss.item(), i)

                running_training_loss += loss.item()

            if True:  # print every 5 mini-batches
                logging.log(logging.INFO, f'[{epoch + 1}, {i + 1:5d}] loss: {running_training_loss / BATCH_SIZE:.3f}')

    logging.log(logging.INFO, 'Finished Training')
    save_model(model, optimizer, epoch, i, loss.item())
    writer.close()
