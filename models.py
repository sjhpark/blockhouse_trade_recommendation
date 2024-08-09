import logging
from transformers import BertModel, BertForSequenceClassification, BertTokenizer

def model_package(model_name:str, num_labels:int=3) -> tuple:
    """
    Load the pre-trained model and tokenizer.
    """
    # turn off the logging for transformers to avoid unnecessary warnings
    # source: https://stackoverflow.com/questions/78827482/cant-suppress-warning-from-transformers-src-transformers-modeling-utils-py
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if "transformers" in logger.name.lower():
            logger.setLevel(logging.ERROR)

    # pre-trained weights
    bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

    return bert_model, tokenizer
