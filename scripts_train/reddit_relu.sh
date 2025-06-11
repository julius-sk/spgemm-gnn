# if [ "$#" -ne 3 ]; then
#   echo "Usage: $0 <model> <gpu> <seed>"
#   exit 1
# fi

# # Assign the input arguments to variables
# model="$1"
# gpu="$2"
# seed="$3"
# export dataset=reddit

# mkdir -p ./log/${dataset}_seed${seed}/
# nohup python -u maxk_gnn_dgl.py --dataset ${dataset} --model ${model} \
#  --hidden_layers 4 --hidden_dim 256 --nonlinear "relu" \
#  --dropout 0.5 --norm --w_lr 0.01 --seed ${seed} \
#  --path experiment/${dataset}_seed${seed}/${model}_relu --epochs 3000 --gpu ${gpu} \
#  > ./log/${dataset}_seed${seed}/${model}_relu.txt &

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <model> <gpu> <seed>"
  exit 1
fi

# Assign the input arguments to variables
model="$1"
gpu="$2"
seed="$3"
export dataset=reddit

# Create log directory
mkdir -p ./log/${dataset}_seed${seed}/

# Record start time
echo "Start time: $(date)" > ./log/${dataset}_seed${seed}/${model}_relu_time.txt

# Run the training with time command
TIMEFORMAT='Total training time: %3R seconds'
(time nohup python -u maxk_gnn_dgl.py --dataset ${dataset} --model ${model} \
 --hidden_layers 4 --hidden_dim 256 --nonlinear "relu" \
 --dropout 0.5 --norm --w_lr 0.01 --seed ${seed} \
 --path experiment/${dataset}_seed${seed}/${model}_relu --epochs 400 --gpu ${gpu} \
 > ./log/${dataset}_seed${seed}/${model}_relu.txt) 2>> ./log/${dataset}_seed${seed}/${model}_relu_time.txt &

# Record end time in the background, waiting for the training process to finish
pid=$!
wait $pid
echo "End time: $(date)" >> ./log/${dataset}_seed${seed}/${model}_relu_time.txt