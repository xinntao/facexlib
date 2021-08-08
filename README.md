# ![icon](assets/icon_small.png) FaceXLib

[![PyPI](https://img.shields.io/pypi/v/facexlib)](https://pypi.org/project/facexlib/)
[![download](https://img.shields.io/github/downloads/xinntao/facexlib/total.svg)](https://github.com/xinntao/facexlib/releases)
[![Open issue](https://isitmaintained.com/badge/open/xinntao/facexlib.svg)](https://github.com/xinntao/facexlib/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/facexlib.svg)](https://github.com/xinntao/facexlib/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/facexlib/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/facexlib/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/publish-pip.yml)
[![gitee mirror](https://github.com/xinntao/facexlib/actions/workflows/gitee-mirror.yml/badge.svg)](https://github.com/xinntao/facexlib/blob/master/.github/workflows/gitee-mirror.yml)

[English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/facexlib) **|** [Gitee码云](https://gitee.com/xinntao/facexlib)

---

**facexlib** aims at providing ready-to-use **face-related** functions based on current STOA open-source methods. <br>
Only PyTorch reference codes are available. For training or fine-tuning, please refer to their original repositories listed below. <br>
Note that we just provide a collection of these algorithms. You need to refer to their original LICENCEs for your intended use.

If facexlib is helpful in your projects, please help to :star: this repo. Thanks:blush: <br>
Other recommended projects: &emsp; :arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) &emsp; :arrow_forward: [GFPGAN](https://github.com/TencentARC/GFPGAN) &emsp; :arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR)

---

## :sparkles: Functions

| Function | Sources  | Original LICENSE |
| :--- | :---:        |     :---:      |
| [Detection](facexlib/detection/README.md) | [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | MIT |
| [Alignment](facexlib/alignment/README.md) |[AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) | Apache 2.0 |
| [Recognition](facexlib/recognition/README.md) | [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) | MIT |
| [Parsing](facexlib/parsing/README.md) | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | MIT |
| [Matting](facexlib/matting/README.md) | [MODNet](https://github.com/ZHKKKe/MODNet) | CC 4.0 |
| [Headpose](facexlib/headpose/README.md) | [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) | Apache 2.0  |
| [Tracking](facexlib/tracking/README.md) |  [SORT](https://github.com/abewley/sort) | GPL 3.0 |
| [Assessment](facexlib/assessment/README.md) | [hyperIQA](https://github.com/SSL92/hyperIQA) | - |
| [Utils](facexlib/utils/README.md) | Face Restoration Helper | - |

## :eyes: Demo and Tutorials

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation

```bash
pip install facexlib
```

### Pre-trained models

It will **automatically** download pre-trained models at the first inference. <br>
If your network is not stable, you can download in advance (may with other download tools), and put them in the folder: `PACKAGE_ROOT_PATH/facexlib/weights`.

## :scroll: License and Acknowledgement

This project is released under the MIT license. <br>

## :e-mail: Contact

If you have any question, open an issue or email `xintao.wang@outlook.com`.
