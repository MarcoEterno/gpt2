from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from src.config import BATCH_SIZE, TRAIN_NUM_EPOCHS
from src.config import CONTEXT_LENGTH
from src.config import DEVICE
from src.config import EMBEDDING_SIZE
from src.config import NUM_DECODERS
from src.config import NUM_HEADS
from src.config import POSITIONAL_ENCODING_COEFFICIENT
from src.config import POSITIONAL_ENCODING_SCALAR
from src.config import VOCAB_SIZE
from src.model.gpt2 import GPT2
from src.utils.dataset import train_ds
from src.utils.tokenizer import tokenize


def lr_rate(step_num, d_model, factor, warmup_steps):
    step_num = max(1, step_num)
    return factor * (
        d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    )


if __name__ == "__main__":
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

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step_num: lr_rate(
            step_num, d_model=512, factor=1, warmup_steps=4000
        ),
    )

    for epoch in range(TRAIN_NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            # print(model(batch))

            inputs = batch['inputs_ids']
            labels = batch['inputs_ids']  # TODO change and add loop

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        print('Finished Training')
