import yaml
import termcolor
import pandas as pd
import torch
from torch.utils.data import Dataset

def color_print(text:str, color:str='green', bold:bool=True) -> None:
    bold = ['bold'] if bold else []
    print(termcolor.colored(text, color, attrs=bold))

# yml loader
def config_load(config_path:str) -> dict:
    """Load the config file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def signal_map(side:str) -> str:
    """
    Map the side (aggressor in trade) to corresponding trading signal.
    This is a naive approach as the side indicates which side of the aggressor intiated the action.
    - A (Ask; Sell aggressor): Sell
    - B (Bid; Buy agressor): Buy
    - N (None; No aggressor): Hold
    """
    if side == 'A':
        return 'Sell'
    elif side == 'B':
        return 'Buy'
    else:
        return 'Hold'
    
def drop_single_value_vars(df:pd.DataFrame) -> pd.DataFrame:
    """Drop columns (variables) that have only one unique value.
    Because they are not useful for training the model."""
    # find indepedent variables that have a single value.
    single_value_vars = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            single_value_vars.append(col)
    color_print(f"Single-value variables to drop from dataset: {single_value_vars}", color='red', bold=False)
    return df.drop(columns=single_value_vars)

def data_preprocess(df:pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for analysis."""
    # Drop columns with single value
    df = drop_single_value_vars(df)
    # Map 'side' to 'signal' (Buy, Sell, Hold) and add it as a new column
    df['signal'] = df['side'].apply(signal_map)
    # replace NaN with 0
    df.fillna(0, inplace=True)
    return df

class TBBO_Dataset(Dataset):
    def __init__(self, df_processed, tokenizer, max_len):
        """
        df_processed is the market data in TBBO Schema (https://databento.com/datasets/XNAS.ITCH) after preprocessing.
        """
        self.data = df_processed
        self.tokenizer = tokenizer
        self.max_len = max_len # max token length
        self.signal2int = {'Buy': 0, 'Sell': 1, 'Hold': 2} # mapping of signal to integer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx] # row of both dependent and independent variables
        row_ind_var = row.drop(['side', 'signal']) # row of independent variables
        
        # ================ Feature Extraction from Independent Variables ================
        # features from independent variables
        var_names = row_ind_var.index.tolist()
        features = row_ind_var[var_names].values # all the independent variables are numerical here

        # convert to string b/c BERT tokenizer expects string
        features = ' '.join([str(feature) for feature in features]) # BERT expects [batch_size, seq_len], so concat all the features into a single string
        
        # Tokenize the text (string) data
        inputs = self.tokenizer(
            features,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract input IDs and attention mask
        input_ids = inputs['input_ids'].squeeze(0) # remove batch dim
        attention_mask = inputs['attention_mask'].squeeze(0) # remove batch dim

        # ================ Target Extraction from Dependent Variable ================

        # target (dependent variable)
        target = row['signal'] # categorical [Buy, Sell, Hold]
        target = self.signal2int[target] # convert to integer
        target = torch.tensor(target, dtype=torch.long)

        # return features, target
        return input_ids, attention_mask, target

