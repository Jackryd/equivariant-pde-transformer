# Attribution

This repository contains Jack Elliot Collier Ryder's MMA440 course project, an adaptation of the official mixed-channel PDE-Transformer architecture for D4-equivariant shallow-water forecasting. It uses the upstream package as a dependency rather than including the upstream source tree.

- **PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations** — Benjamin Holzschuh, Qiang Liu, Georg Kohl, and Nils Thuerey. [Source](https://github.com/tum-pbs/pde-transformer). The license file supplied with the source used for this project is preserved verbatim in [licenses/pde-transformer-LICENSE.txt](licenses/pde-transformer-LICENSE.txt).
- **Vanilla Group Equivariant Vision Transformer: Simple and Effective** — Jiahong Fu, Qi Xie, Deyu Meng, and Zongben Xu. [Paper](https://arxiv.org/abs/2602.08047). The paper-oriented variant adapts its group-mixing maps, pair-orbit positional encoding, and canonical resampling constructions to the PDE-Transformer graph.
- **PDEBench** supplies the shallow-water benchmark dataset, downloaded separately. [Source](https://github.com/pdebench/PDEBench).
- **neuraloperator** supplies the optional FNO implementation. [Source](https://github.com/neuraloperator/neuraloperator).

The D4 adaptations, experiment code, and project presentation are the course-project contribution. The PDF documents the original experiments; the GIFs are existing training-trajectory illustrations. Repository cleanup does not represent a new training run.
