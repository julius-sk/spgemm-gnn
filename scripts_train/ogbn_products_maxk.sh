# Check if the correct number of arguments are provided
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <k> <seed> <gpu> <model> [cache-strategy] [cache-ratio]"
  echo "Cache strategies: none, direct, static-outd, static-pres, fifo, lru"
  echo "Cache ratio: Value between 0 and 1 (default: 0.3)"
  exit 1
fi

# Assign the input arguments to variables
k="$1"
seed="$2"
gpu="$3"
model="$4"
cache_strategy="${5:-none}"  # Default to 'none' if not provided
cache_ratio="${6:-0.3}"      # Default to 0.3 if not provided

export dataset=ogbn-products

if [ "$model" == "sage" ]; then
    selfloop=""
else
    selfloop=--selfloop
fi

# Create directory for logs
mkdir -p ./log/${dataset}_seed${seed}/

# Prepare the output filename with cache strategy info
if [ "$cache_strategy" == "none" ]; then
    output_suffix="${model}_max${k}"
else
    output_suffix="${model}_max${k}_${cache_strategy}_${cache_ratio}"
fi

# Run the command with the appropriate cache strategy parameters
nohup python -u maxk_gnn_dgl.py --dataset ${dataset} --model ${model} ${selfloop} \
 --hidden_layers 3 --hidden_dim 256 --nonlinear "maxk" --maxk ${k} \
 --dropout 0.5 --norm --w_lr 0.003 --seed ${seed} \
 --path experiment/${dataset}_seed${seed}/${output_suffix} \
 --epochs 500 --gpu ${gpu} \
 --cache-strategy ${cache_strategy} --cache-size-ratio ${cache_ratio} <<< "y" \
 > ./log/${dataset}_seed${seed}/${output_suffix}.txt &
