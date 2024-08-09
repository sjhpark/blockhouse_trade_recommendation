# Create docker image (name: tbbo_trading) using docker/dockerfile
sudo docker build docker -t tbbo_trading

# Run docker image to create docker container
sudo docker run -it --network=host --gpus all -v $PWD:/workspace tbbo_trading