<div align="center">

# SimMOT

## SimMOT: Similarity-guided Trajectory Aggregation for Online Multi-Object Tracking

:e-mail: Primary contact: huangyuanzhou@bupt.edu.cn

</div>

## Abstract

Learning discriminative trajectory representations is crucial for improving association in online Multi-Object Tracking (MOT). While recent methods can leverage long-term historical information to enhance trajectory representations, they often overlook the presence of less informative or noisy features in trajectories and rely on a single long-term memory. This can compromise association performance in complex scenarios, particularly when targets experience frequent motion changes or occlusions. In this paper, we propose SimMOT, a Transformer-based model with a novel trajectory aggregation method that dynamically refines trajectory representations to address these challenges in long-term tracking. Guided by learned intra-trajectory similarity, which helps capture temporal variations within trajectories, SimMOT learns trajectory representations in a hierarchical manner. It dynamically selects salient features at the frame level to reduce noise, and learns distinct long-term memories at the trajectory level to adapt to diverse motion patterns. By aggregating salient features and long-term memories, the obtained representations retain both low-level salient temporal details and high-level motion patterns, thereby improving association performance. Experimental results demonstrate the effectiveness of our method, achieving competitive performance on the DanceTrack, SportsMOT and BFT benchmarks.

## Upcoming

The initial implementation and pretrained weights for DanceTrack are currently available. We are actively improving the repository and preparing additional resources.

- [ ] **Documentation refinement**: Add more detailed documentation for configuration, data preparation, and usage.
- [ ] **Code cleanup**: Clean up legacy experiment scripts and unused configurations; add unit tests.
- [ ] **Results on more datasets**: Release training configs and results on SportsMOT and BFT.
- [ ] **Ablation guide**: Provide instructions and configs for running ablation experiments.

## Quick Start

You can quickly get started with SimMOT through the following steps:

1. **Environment setup**: Prepare the Python environment and compile CUDA ops. See [INSTALL.md](./docs/INSTALL.md).
2. **Folder creation**: Create directories for saving datasets, checkpoints, outputs, and scripts. See [PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md).
3. **Dataset preparation**: Download datasets. See [DATASET.md](./docs/DATASET.md).
4. **Weight files preparation**: Download pretrained weights and checkpoints. See [MODEL_ZOO.md](./docs/MODEL_ZOO.md).
5. **Training and inference**: Execute training and inference commands. See [GET_STARTED.md](./docs/GET_STARTED.md).

## Acknowledgements

This project is built upon [MOTIP](https://github.com/MCG-NJU/MOTIP). We thank the contributors of this great codebase.

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{huang2026simmot,
  title={SimMOT: Similarity-guided Trajectory Aggregation for Online Multi-Object Tracking},
  author={Huang, Yuanzhou and Pei, Songwei and Zeng, Rui and Wang, Shuhuai and Liu, Bingfeng and Li, Qian and Wang, Shangguang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  doi={10.1109/TCSVT.2026.3735893}
}
```
