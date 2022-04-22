# assumes conda is being used and number of nodes is 3
# takes two arguments: ip_address and rank in that order
python main.py --master-ip $1 --num-nodes 3 --rank $2