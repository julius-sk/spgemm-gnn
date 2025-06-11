# # Check if the correct number of arguments are provided
# if [ "$#" -ne 4 ]; then
#   echo "Usage: $0 <k> <seed> <gpu> <model>"
#   exit 1
# fi

# # Assign the input arguments to variables
# k="$1"
# seed="$2"
# gpu="$3"
# model="$4"
# export dataset=reddit

# mkdir -p ./log/${dataset}_seed${seed}/
# nohup python -u maxk_gnn_dgl.py --dataset ${dataset} --model ${model} \
#  --hidden_layers 4 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
#  --dropout 0.5 --norm --w_lr 0.01 --seed ${seed} \
#  --path experiment/${dataset}_seed${seed}/${model}_max${k} --epochs 3000 --gpu ${gpu} \
#  > ./log/${dataset}_seed${seed}/${model}_max${k}.txt &

# Check if the correct number of arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <k> <seed> <gpu> <model>"
  exit 1
fi

# Assign the input arguments to variables
k="$1"
seed="$2"
gpu="$3"
model="$4"
export dataset=reddit

# Create log directory
mkdir -p ./log/${dataset}_seed${seed}/

# Record start time
echo "Start time: $(date)" > ./log/${dataset}_seed${seed}/${model}_max${k}_time.txt

# Run the training with time command
TIMEFORMAT='Total training time: %3R seconds'
(time nohup python -u maxk_gnn_dgl.py --dataset ${dataset} --model ${model} \
 --hidden_layers 4 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
 --dropout 0.5 --norm --w_lr 0.01 --seed ${seed} \
 --path experiment/${dataset}_seed${seed}/${model}_max${k} --epochs 400 --gpu ${gpu} \
 > ./log/${dataset}_seed${seed}/${model}_max${k}.txt) 2>> ./log/${dataset}_seed${seed}/${model}_max${k}_time.txt &

# Record end time in the background, waiting for the training process to finish
pid=$!
wait $pid
echo "End time: $(date)" >> ./log/${dataset}_seed${seed}/${model}_max${k}_time.txt