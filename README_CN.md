# ![icon](assets/icon_small.png) FaceXLib

[English](README.md) **|** [简体中文](README_CN.md)

---

`facexlib` is a **pytorch-based** library for **face-related** functions, such as detection, alignment, recognition, tracking, utils for face restorations, *etc*.
It only provides inference (without training).
This repo is based current STOA open-source methods (see [more details](#Functions)).

## :eyes: Demo

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## :sparkles: Functions

| Function | Description  | Reference |
| :--- | :---:        |     :---:      |
| Detection | ([More details](detection/README.md) | [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) |
| Alignment | ([More details](alignment/README.md) | [AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) |
| Recognition | ([More details](recognition/README.md) | [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) |
| Tracking | ([More details](tracking/README.md) | [SORT](https://github.com/abewley/sort) |
| Utils | ([More details](utils/README.md)) | |

## :scroll: License and Acknowledgement

This project is released under the MIT license. <br>

## :e-mail: Contact

If you have any question, open an issue or email `xintao.wang@outlook.com`.
