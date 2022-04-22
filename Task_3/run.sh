python main.py
torch-model-archiver --model-name vgg --version 1.0 --model-file model.py --serialized-file vgg_cnn.pt --handler vgg_handler.py
mv vgg.mar model_store/
torchserve --start --model-store model_store --models vgg=vgg.mar