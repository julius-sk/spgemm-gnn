#!/bin/bash
set -e

DATASETS=(ogbn-products ogbn-arxiv)
MODELS=(sage gcn gin)
K_VALUES=(16 32 64 128)

run_experiment() {
    local dataset=$1 model=$2 k=$3 use_maxk=$4 gpu=${5:-0}
    local exp_id="${dataset}_${model}_k${k}_maxk${use_maxk}"
    
    mkdir -p logs
    
    local cmd="python maxk_gnn_integrated.py --dataset $dataset --model $model --maxk $k --gpu $gpu --nonlinear relu"
    [[ "$use_maxk" == "true" ]] && cmd="$cmd --use_maxk_kernels"
    
    export CUDA_VISIBLE_DEVICES=$gpu
    timeout 7200 $cmd > "logs/${exp_id}.log" 2>&1 || echo "FAILED: $exp_id"
}

run_all() {
    local kernels_available=$(python -c "try: import maxk_cuda_kernels; print('true')" 2>/dev/null || echo "false")
    
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            
            # ReLU case - no k value needed
            run_experiment "$dataset" "$model" "false" "relu" "" 0
            [[ "$kernels_available" == "true" ]] && run_experiment "$dataset" "$model" "true" "relu" "" 0
            
            # MaxK case - run with all k values
            for k in "${K_VALUES[@]}"; do
                run_experiment "$dataset" "$model" "false" "maxk" "$k" 0
                [[ "$kernels_available" == "true" ]] && run_experiment "$dataset" "$model" "true" "maxk" "$k" 0
            done
            
        done
    done
}

case "${1:-help}" in
    all) run_all ;;
    *) echo "Usage: $0 all" ;;
esac
