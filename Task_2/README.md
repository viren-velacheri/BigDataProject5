## Task 2 Instructions

The command to run task 2 is as follows:

<b> ./run.sh $ip_address$ $rank$ </b>

Example: ./run.sh "10.10.1.1" 0

This command will need to be run across all 3 machines and the only argument that would change is the rank passed in. Please ensure that 0 is passed in for rank on the main machine and 1 and 2 are passed in when running on the other machines. This will train a model using distributed data parallel training. It will print the loss value after every 20 iterations. The current code only runs the training loop for 40 iterations and reports the average time per iteration for the first 40 iterations (disregarding the first iteration) to collect data for the report as specified in the deliverables.