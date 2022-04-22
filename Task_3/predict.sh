# run script used to check the prediction of kitten.jpg for the vgg_model
# stops TorchServer after the prediction
curl http://127.0.0.1:8081/models
curl http://127.0.0.1:8080/predictions/vgg -T kitten.jpg
torchserve --stop