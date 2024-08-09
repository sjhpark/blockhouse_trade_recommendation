import yaml
import termcolor
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

def color_print(text:str, color:str='green', bold:bool=True) -> None:
    bold = ['bold'] if bold else []
    print(termcolor.colored(text, color, attrs=bold))

# yml loader
def config_load(config_path:str) -> dict:
    """Load the config file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def round_unix(time:int) -> int:
    """
    Round the unix timestamp (in nanoseconds) to be in seconds.
    This is for passing in the correct time format for the Pytorch Forecasting's TimeSeriesDataSet object.
    """
    return int(time / 1e9)

def signal_map(side:str) -> str:
    """
    Map the side (aggressor in trade) to corresponding trading signal.
    This is a naive approach as the side indicates which side of the aggressor intiated the action.
    - A (Ask; Sell aggressor): Sell
    - B (Bid; Buy agressor): Buy
    - N (None; No aggressor): Hold
    """
    if side == 'A':
        return 0 # 'Sell'
    elif side == 'B':
        return 1 # 'Buy'
    else:
        return 2 # 'Hold'
    
def get_single_value_vars(df:pd.DataFrame) -> list:
    """Get the columns that have only one value."""
    return [col for col in df.columns if df[col].nunique() == 1]

def data_preprocess(df:pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for analysis."""
    # round the unix timestamp to seconds
    df['ts_recv'] = df['ts_recv'].apply(round_unix)
    # map 'side' to 'signal' (0 for Buy, 1 for Sell, 2 for Hold) and add it as a new column
    df['signal'] = df['side'].apply(signal_map)
    # drop side column
    df.drop(columns='side', inplace=True)
    # drop ts_event column
    df.drop(columns='ts_event', inplace=True)
    # replace NaN with 0
    df.fillna(0, inplace=True)
    return df

def create_dataset(df_processed:pd.DataFrame,
            max_encoder_length:int, max_prediction_length:int) -> TimeSeriesDataSet:
    """Create a TimeSeriesDataSet object from the DataFrame."""
    target_var = 'signal'
    static_vars = get_single_value_vars(df_processed)
    static_categorical_vars = [col for col in static_vars if df_processed[col].dtype == 'object']
    static_numerical_vars = [col for col in static_vars if col not in static_categorical_vars]
    dynamic_numerical_vars = [col for col in df_processed.columns if col not in static_vars and col != target_var]

    dataset = TimeSeriesDataSet(
        df_processed,
        time_idx="ts_recv", # timestamp
        target=target_var, # target variable
        group_ids=["instrument_id"],
        max_encoder_length=max_encoder_length,  # number of historical time steps to use for making predictions
        max_prediction_length=max_prediction_length,  # number of future time steps the model is expected to predict
        static_categoricals=static_categorical_vars, # categorical single-value variables
        static_reals=static_numerical_vars, # static real (numerical) variables
        time_varying_known_reals=dynamic_numerical_vars, # Dynamic real (numerical) variables
        time_varying_unknown_reals=[],
        allow_missing_timesteps=True, # Handle time difference between steps > 1
    )
    return dataset


