import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from models import model_package
from utils import config_load, data_preprocess, TBBO_Dataset, color_print

if __name__ == '__main__':

    # configuration
    config = config_load(config_path='config.yml')
    data_path = config['data']['path']
    model_name = config['model']['name']
    max_len = config['model']['max_len']
    batch = config['train']['batch']
    epochs = config['train']['epochs']
    lr = config['train']['lr']
    device = torch.device(config['model']['device'])
    for key, value in config.items():
        color_print(f"{key}: {value}", color='yellow', bold=False)
    
    # number of classes (Buy, Sell, Hold)
    num_classes = 3
    color_print(f"Numbere of Classes: {num_classes}", color='yellow', bold=False)

    # model package
    model, tokenizer = model_package(model_name, num_labels=num_classes)
    model.to(device)
    color_print(f"Model: {model_name}", color='blue', bold=True)

    # data preparation
    df = pd.read_csv(data_path)
    df_processed = data_preprocess(df)

    # 80/10/10 train/test/validation split
    train, temp = train_test_split(df_processed, test_size=0.2, random_state=1)
    test, val = train_test_split(temp, test_size=0.5, random_state=1)

    # dataset
    train_dataset = TBBO_Dataset(train, tokenizer, max_len)
    test_dataset = TBBO_Dataset(test, tokenizer, max_len)
    val_dataset = TBBO_Dataset(val, tokenizer, max_len)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    color_print(f"Optimizer: {optimizer.__class__.__name__}", color='blue', bold=False)

    # loss function
    criterion = nn.CrossEntropyLoss()
    color_print(f"Loss Function: {criterion.__class__.__name__}", color='blue', bold=False)

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (input_ids, attention_mask, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss
            loss = criterion(outputs.logits, targets)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, attention_mask, targets in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
