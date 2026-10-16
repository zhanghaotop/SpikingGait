# SpikingGait

Official PyTorch implementation of **SpikingGait: A Structural-to-Volumetric
Spike-Driven Backbone for Gait Recognition**, accepted by ACM Multimedia 2026.
This codebase is built on [OpenGait](https://github.com/ShiqiYu/OpenGait).

[[Paper](https://doi.org/10.1145/3767308.3836007)]

## Highlights

- A structural-to-volumetric hierarchy for silhouette gait sequences.
- MultiSpike activations throughout the spike-driven backbone.
- Lightweight temporal fusion in the structural stage.
- An OpenGait-compatible gait recognition head.
- A SUSTech1K training and evaluation configuration.

## Repository contents

```text
.
├── opengait/modeling/models/spikinggait.py
├── configs/spikinggait/spikinggait_sustech1k.yaml
├── datasets/SUSTech1K/SUSTech1K.json
├── CITATION.cff
├── environment.yml
└── requirements.txt
```

This repository currently provides only the SUSTech1K configuration and its
OpenGait train/test partition (250/800 identities). The dataset, preprocessed
subject data, training logs, and checkpoints are not distributed.

## Installation

### 1. Clone OpenGait

```bash
git clone https://github.com/ShiqiYu/OpenGait.git
```

Install the OpenGait dependencies by following its official documentation.

### 2. Prepare the environment

From the SpikingGait repository:

```bash
conda env create -f environment.yml
conda activate spikinggait
```

Alternatively, in an existing compatible PyTorch environment:

```bash
pip install -r requirements.txt
```

Make sure the installed PyTorch build is compatible with the local CUDA
runtime and GPUs.

## Training

```bash
sh train.sh
```

## Testing

```bash
sh test.sh
```

## Citation

If this repository is useful in your research, please cite SpikingGait and
OpenGait:

```bibtex
@inproceedings{zhang2026spikinggait,
  author    = {Hao Zhang and Jie Zhang and Tianyu Jiang and Zheng Fang and Chunsheng Hua},
  title     = {SpikingGait: A Structural-to-Volumetric Spike-Driven Backbone for Gait Recognition},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3836007}
}
```

Please use the citation requested by the
[OpenGait repository](https://github.com/ShiqiYu/OpenGait) for the underlying
framework.

## Acknowledgements

This implementation is built on OpenGait. We thank its authors and contributors
for providing the gait recognition framework.

MultiSpike is credited in the model source to its original paper and official
implementation.
