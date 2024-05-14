# This module implements some modalities to chat with the model
from transformers import GPT2Tokenizer

from src.config import DEVICE
from src.model.gpt2 import GPT2


def respond_to_user_input(user_input: str, model: GPT2, tokenizer: GPT2Tokenizer, NUM_RETURN_SEQUENCES=None,
                          MAX_LENGTH=None):
    """
    Respond to user input using the model.

    Args:
        user_input (str): The user input to respond to.
        model (GPT2): The model to use for generating the response.
        tokenizer (GPT2Tokenizer): The tokenizer to use for tokenizing the input.

    Returns:
        str: The response to the user input.
    """
    user_input = tokenizer.encode(user_input, return_tensors="pt").to(DEVICE)
    output = model.generate(
        user_input,
        max_length=MAX_LENGTH,
        num_return_sequences=NUM_RETURN_SEQUENCES,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        do_sample=DO_SAMPLE,
        top_k=TOP_K,
        top_p=TOP_P,
        temperature=TEMPERATURE
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response