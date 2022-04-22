## Task 1 Instructions

The command to run task 1 is as follows:

<b> ./run.sh </b>

This will train a model on VGG-11 network with Cifar10 dataset. It will print the loss value after every 20 iterations. The training will run for a total of 1 epoch (i.e., until every example has been seen once) with batch size 256. The current code only runs the training loop for 40 iterations and reports the average time per iteration for the first 40 iterations (disregarding the first iteration) to collect data for the report as specified in the deliverables.