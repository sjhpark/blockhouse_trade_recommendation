import os
import time
import pandas as pd
from models import model_package
from utils import config_load, color_print, data_preprocess, create_dataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_forecasting.metrics import CrossEntropy, QuantileLoss

if __name__ == '__main__':

    # configuration
    config = config_load(config_path='config.yml')
    data_path = config['path']['data']
    checkpoint_path = config['path']['checkpoint']
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    device = config['model']['device']
    hidden_size = config['model']['hidden_size']
    hidden_continuous_size = config['model']['hidden_continuous_size']
    attention_head_size = config['model']['attention_head_size']
    max_encoder_length = config['model']['max_encoder_length']
    max_prediction_length = config['model']['max_prediction_length']

    num_workers = config['train']['num_workers']
    batch = config['train']['batch']
    epochs = config['train']['epochs']
    optimizer = config['train']['optimizer']
    lr = config['train']['lr']
    dropout = config['train']['dropout']

    for key, value in config.items():
        color_print(f"{key}: {value}", color='yellow', bold=False)
    
    # load data
    df = pd.read_csv(data_path)
    df_processed = data_preprocess(df)
    color_print(f"Data loaded and processed.", color='light_green', bold=False)

    # create dataset
    dataset = create_dataset(df_processed, max_encoder_length, max_prediction_length)
    color_print(f"Dataset created.", color='light_green', bold=False)

    # dataloaders
    train_dataloader = dataset.to_dataloader(train=True, batch_size=batch, num_workers=num_workers)
    val_dataloader = dataset.to_dataloader(train=False, batch_size=batch, num_workers=num_workers)
    color_print(f"Dataloaders created.", color='light_green', bold=False)

    # loss function
    criterion = QuantileLoss()
    color_print(f"Loss Function: {criterion.__class__.__name__}", color='cyan', bold=False)

    # optimizer
    optimizer = optimizer
    color_print(f"Optimizer: {optimizer.__class__.__name__}", color='cyan', bold=False)

    # model
    model = model_package(dataset, criterion, optimizer,
                        lr, dropout, 
                        hidden_size, hidden_continuous_size, attention_head_size)
    model_name = model.__class__.__name__
    color_print(f"Model: {model_name}", color='magenta', bold=True)

    # configure model checkpoint
    curr_time = time.strftime("%m%d%y_%H%M")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=f"checkpoint_{model_name}_{curr_time}",
        save_top_k=1, # save only the best model
        monitor="val_loss", # metric to monitor
        mode="min", # mode for the monitored metric ('min' for loss)
        save_weights_only=True, # Save only the model weights
    )
    
    # training
    color_print(f"Training started...", color='blue', bold=True)
    trainer = Trainer(max_epochs=epochs, 
                    accelerator=device, devices=1 if device=='cuda' else 0,
                    callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, val_dataloader)
    color_print(f"Training completed. Model saved at: {checkpoint_path}", color='blue', bold=True)
