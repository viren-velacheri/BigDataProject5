## Task 3 Instructions

NOTE: Make sure to run "pip3 install captum" before running any of the following commands. It is a dependency and the prediction will hang if it is not installed prior to starting the server.

The command to run task 3 is as follows:

<b> ./run.sh </b>

This will train a model on VGG-11 network with Cifar10 dataset. It will print the loss value after every 20 iterations. The training will run for a total of 1 epoch (i.e., until every example has been seen once) with batch size 256. The run script will also save the model into vgg_cnn.pt and start the TorchServe server.

From here, you should create a new session on the same machine and run the following command:

<b> ./predict.sh </b>

The run script will check if the vgg model is loaded, check the prediction of kitten.jpg for the vgg_model, and then stop TorchServer.

NOTE: We are assuming that kitten.jpg is in the same directory as where the prediction run script (predict.sh) is being run.
