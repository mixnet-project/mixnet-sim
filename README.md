# Simulation of MixNet

This folder contains simulation code of MixNet. mixnet-flexflow implements hybrid parallelism task graph generation, while mixnet-htsim implements packet-level network simulation and mixnet runtime reconfiguration logic for thorough evaluation of large-scale MoE training. 

## Simulation Process

Each subfolder contains detailed guideline on how to build and run the simulator. A general simulation process contains two steps:

1. Run mixnet-flexflow simulator on the designd model, and get a **taskgraph** in FlatBuffer. The taskgraph is exported via the `--taskgraph` flag in FlexNet packet simulator.
2. Run mixnet-htsim simulator with the taskgraph acquired from FlexNet simulator to evaluate the performance.

## End-to-End Workflow

### Step 1: Generate Task Graph (fbuf)

**⚠️ Important: Update the directory path in the script to match your local setup.**

First, generate the task graph fbuf files using mixnet-flexflow. The following example generates task graphs for Mixtral-8x22B model with different microbatch sizes:

```bash
# mixtral8x22B: microbatch sizes
declare -a microbatchsize=(8)

# ⚠️ UPDATE THIS PATH to your FlexFlow directory
dir="/usr/wkspace/mixnet/FlexFlow-master"
cd "$dir" || exit 1

# full model
for mb in "${microbatchsize[@]}"; do
    ./build/examples/cpp/mixture_of_experts/moe -ll:gpu 1 -ll:fsize 31000 -ll:zsize 24000 \
    --budget 20 --only-data-parallel --batchsize 128 --microbatchsize "$mb" \
    --train_dp 2 --train_tp 8 --train_pp 8 --num-layers 56 --embedding-size 1024 \
    --expert-hidden-size 16384 --hidden-size 6144 --num-heads 32 --sequence-length 4096 \
    --topk 2 --expnum 8
    
    mv output.txt ./results
    cd ./results
    mv output.txt mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.txt
    mv taskgraph.fbuf mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf
    cd ..
done
```

This will generate fbuf files in the `results` directory (e.g., `mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf`).

### Step 2: Run Network Simulation

**⚠️ Important: Update both directory paths in the script to match your local setup.**

After generating the task graph fbuf files, run the network simulation using mixnet-htsim:

```bash
declare -a bw=(100)
declare -a microbatchsize=(8)
declare -a rdelay=(25)
declare -a workloads=("num_global_tokens_per_expert")

# ⚠️ UPDATE THIS PATH to your mixnet-htsim datacenter directory
dir="/usr/wkspace/mixnet/flexnet-sim-refactor/src/clos/datacenter"

# ⚠️ UPDATE THIS PATH to where your fbuf files were generated (Step 1)
new_fbuf_dir="/usr/wkspace/mixnet/FlexFlow-master/results"

cd "$dir" || exit 1

for mb in "${microbatchsize[@]}"; do
    for b in "${bw[@]}"; do
        for r in "${rdelay[@]}"; do
            for workload in "${workloads[@]}"; do
                mkdir -p ./logs/mixtral8x22B_${mb}_mixnet_${b}_${workload}/
                ./htsim_tcp_mixnet -simtime 3600.1 \
                -flowfile ${new_fbuf_dir}/mixtral8x22B_dp2_tp8_pp8_ep8_${mb}.fbuf \
                -speed $((b*1000)) \
                -ocs_file nwsim_ocs_${b}.txt \
                -ecs_file nwsim_ecs_${b}.txt \
                -nodes 1024 \
                -ssthresh 10000 \
                -rtt 1000 \
                -q 10000 \
                -dp_degree 2 \
                -tp_degree 8 \
                -pp_degree 8 \
                -ep_degree 8 \
                -rdelay $r \
                -weightmatrix ../../../test/${workload}.txt \
                -logdir ./logs/mixtral8x22B_${mb}_mixnet_${b}_${workload}/ \
                > ./logs/mixtral8x22B_${mb}_mixnet_${b}_${workload}/output.log 2>&1 &
            done
        done
    done
done
```

The simulation results will be stored in the `logs/` directory with separate folders for each configuration.

### Key Parameters

- **Microbatch size**: Controls the size of microbatches for training
- **Bandwidth (`bw`)**: Network bandwidth in Gbps (e.g., 100 for 100Gbps)
- **Reconfiguration delay (`rdelay`)**: Delay for network reconfiguration in microseconds
- **Workload**: Weight matrix for expert assignment (e.g., `num_global_tokens_per_expert`)
- **Parallelism degrees**: Data parallel (dp), tensor parallel (tp), pipeline parallel (pp), expert parallel (ep)

