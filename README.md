# blockhouse_trade_recommendation
Trade Recommendation via Transformer using market data with TBBO (Trade with Best Bid and Offer) schema.

## Download Data
```bash
wget https://github.com/Blockhouse-Repo/Blockhouse-Work-Trial/blob/d9c4cd5bd3edc10b053cefab82cc66e2f9753ef6/xnas-itch-20230703.tbbo.csv
```

## Build Docker Image and Run Docker Container
This docker setup installs CUDA 11 development kit for Ubuntu 20.04.
This setup was tested with NVIDIA GeForce GTX 1650 GPU which has Turing architecture (sm_75). 
If you have different NVIDIA GPU architecture (e.g., Ampere for sm_8x, Ada Lovelace for sm_89), you may need to adjust the `docker/dockerfile` accordingly by installing the supported CUDA version.
```bash
sudo sh build_run_docker.sh
```

## At the interactive shell of the Docker Container
```bash
cd workspace
python3 train.py
```

## Configuration
You can adjust configurations (paths, model, training) in `config.yml'.