import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from utils import color_print

def model_package(training:TimeSeriesDataSet, criterion:nn.Module, optimizer:str,
                  lr:float, dropout:float,
                  hidden_size:int, hidden_continuous_size:int, 
                  attention_head_size:int, ) -> TemporalFusionTransformer:
    """
    Load the TFT model.
    Reference: # https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
    """
    model = TemporalFusionTransformer.from_dataset(
        training,
        loss=criterion,
        optimizer=optimizer,
        learning_rate=lr,
        dropout=dropout,  
        hidden_size=hidden_size, 
        hidden_continuous_size=hidden_continuous_size, 
        attention_head_size=attention_head_size, 
    )
    color_print(f"Number of parameters in network: {model.size()/1e3:.1f}k", color='magenta', bold=False)
    return model