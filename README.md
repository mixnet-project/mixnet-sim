# MixNet Simulation

[![SIGCOMM 2025](https://img.shields.io/badge/SIGCOMM-2025-blue)](https://doi.org/10.1145/3718958.3750465)

This repository contains the simulation code for **MixNet**, a runtime reconfigurable optical-electrical fabric for distributed Mixture-of-Experts (MoE) training, published at ACM SIGCOMM 2025.

**MixNet** addresses the unique networking challenges in large-scale MoE training by dynamically reconfiguring network topology to match the dynamic, fine-grained, and skewed all-to-all communication patterns inherent in MoE workloads. By combining optical circuit switching with electrical packet switching, MixNet achieves near-ideal performance compared to classical electrical Fat-tree architectures, while reducing the networking cost by ~2x.

## Simulation Framework

The simulation framework consists of two main components:

- **[mixnet-flexflow](./mixnet-flexflow/)**: Generates hybrid parallelism task graphs for MoE models
- **[mixnet-htsim](./mixnet-htsim/)**: Performs packet-level network simulation and evaluates MixNet's runtime reconfiguration logic for large-scale MoE training

## Overview

The simulation enables end-to-end evaluation of MixNet's performance by combining task graph generation with packet-level network simulation. Each subfolder contains detailed guidelines on how to build and run the respective simulator.

### Simulation Workflow

The simulation process consists of two steps:

1. **Task Graph Generation**: Run the `mixnet-flexflow` simulator on the target MoE model to generate a task graph in FlatBuffer format (`.fbuf` file). The task graph is exported via the `--taskgraph` flag.
2. **Network Simulation**: Run the `mixnet-htsim` packet-level simulator with the generated task graph to evaluate network performance and MixNet's runtime reconfiguration behavior.

## Prerequisites

Before running the simulations, ensure you have:

- fetch the submodules: `git submodule update --init --recursive`
- Built and configured `mixnet-flexflow` (see [mixnet-flexflow](./mixnet-flexflow/) for build instructions)
- Built and configured `mixnet-htsim` (see [mixnet-htsim](./mixnet-htsim/) for build instructions)
- Sufficient storage space for simulation results
- Updated directory paths in the scripts to match your local environment

## Getting Started

### Step 1: Generate Task Graph

**Important**: Update the directory path in the script to match your local setup.

Generate the task graph FlatBuffer (`.fbuf`) files using `mixnet-flexflow`. The following example generates task graphs for the Mixtral-8x22B model with different microbatch sizes:

```bash
# mixtral8x22B: microbatch sizes
declare -a microbatchsize=(8)

# UPDATE THIS PATH to your FlexFlow directory in your local environment
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

This will generate `.fbuf` files in the `results` directory (e.g., `mixtral8x22B_dp2_tp8_pp8_ep8_8.fbuf`).

**Pre-generated Task Graphs**: For your convenience, we provide pre-generated task graph files for the Mixtral-8x22B model in the [Onedrive Link](https://1drv.ms/f/c/e53fc5b825c479a9/EmNMc-aekbFPq0PGvzASJtIBtysORTnPJySwkzCy_5W-fQ?e=yTlZQI).

**Other Models**: To generate task graphs for other MoE models, modify the corresponding parameters in the above script. For example, see the [Mixtral-8x7B configuration](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) for reference.

### Step 2: Run Network Simulation

**Important**: Update both directory paths in the script to match your local setup.

After generating the task graph files, run the network simulation using `mixnet-htsim`. The following example runs the simulation for the Mixtral-8x22B model with the following configuration:

- **Network bandwidth**: 100 Gbps
- **Microbatch size**: 8
- **Reconfiguration delay**: 25 μs
- **Workload**: `num_global_tokens_per_expert`
- **Cluster size**: 1024 GPUs

```bash
declare -a bw=(100)
declare -a microbatchsize=(8)
declare -a rdelay=(25)
declare -a workloads=("num_global_tokens_per_expert")

# UPDATE THIS PATH to your mixnet-htsim datacenter directory
dir="/usr/wkspace/mixnet/flexnet-sim-refactor/src/clos/datacenter"

# UPDATE THIS PATH to where your fbuf files were generated (Step 1)
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

**Quick Testing**: For faster turnaround times during development, you can run single-stage tests. See the [mixnet_scripts](https://github.com/mixnet-project/mixnet-htsim/tree/main/mixnet_scripts) for more details.

## Configuration Parameters

The simulation can be configured using the following key parameters:

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| **Microbatch size** | Number of samples per microbatch | 8, 16, 32 |
| **Bandwidth (`bw`)** | Network bandwidth in Gbps | 100, 200, 400, 800 |
| **Reconfiguration delay (`rdelay`)** | Network reconfiguration delay in milliseconds (ms) | 25, 50, 100 |
| **Workload** | Expert assignment weight matrix | A workload file |
| **Data parallel degree (`dp`)** | Data parallelism degree | 2, 4, 8 |
| **Tensor parallel degree (`tp`)** | Tensor parallelism degree | 4, 8, 16 |
| **Pipeline parallel degree (`pp`)** | Pipeline parallelism degree | 4, 8, 16 |
| **Expert parallel degree (`ep`)** | Expert parallelism degree | 8, 16, 32 |

## Citation

If you use MixNet in your research, please cite our SIGCOMM 2025 paper:

```bibtex
@inproceedings{10.1145/3718958.3750465,
    author = {Liao, Xudong and Sun, Yijun and Tian, Han and Wan, Xinchen and Jin, Yilun and Wang, Zilong and Ren, Zhenghang and Huang, Xinyang and Li, Wenxue and Tse, Kin Fai and Zhong, Zhizhen and Liu, Guyue and Zhang, Ying and Ye, Xiaofeng and Zhang, Yiming and Chen, Kai},
    title = {MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training},
    year = {2025},
    url = {https://doi.org/10.1145/3718958.3750465},
    doi = {10.1145/3718958.3750465},
    booktitle = {Proceedings of the ACM SIGCOMM 2025 Conference},
    pages = {554–574},
}
```

## Acknowledgments

This work builds upon the simulation framework from [TopoOpt](https://github.com/hipersys-team/TopoOpt). We thank the open-source community for their valuable contributions.

## Contact

For questions, issues, or collaboration opportunities, please contact:

- **Xudong Liao**: [xudong.liao.cs@gmail.com](mailto:xudong.liao.cs@gmail.com) | [Website](https://xudongliao.github.io/)
- **Yijun Sun**: [yijun.sun@connect.ust.hk](mailto:yijun.sun@connect.ust.hk)

## License

Please refer to the individual component repositories for licensing information.
