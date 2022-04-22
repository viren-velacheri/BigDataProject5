curl http://127.0.0.1:8081/models
curl http://127.0.0.1:8080/predictions/vgg -T kitten.jpg
torchserve --stop