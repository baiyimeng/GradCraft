# GradCraft
[![arXiv](https://img.shields.io/badge/arXiv-2407.19682-red.svg)](https://arxiv.org/abs/2407.19682)
This is the pytorch implementation of our paper at KDD 2024:
> [GradCraft: Elevating Multi-task Recommendations through Holistic Gradient Crafting](https://arxiv.org/abs/2407.19682)
> 
> Yimeng Bai, Yang Zhang, Fuli Feng, Jing Lu, Xiaoxue Zang, Chenyi Lei, Yang Song.

## Usage
### Data
The experimental datasets are available for download via the link provided in the file located at `/data/download.txt`.
### Training & Evaluation
This project is built on top of [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch).

To use `ray.tune` for hyperparameter tuning in a multi-task setting, you must replace the original `experiment_analysis.py` with the customized version provided in this repo.
```
python main.py
```
## Citation
```
@inproceedings{GradCraft,
author = {Bai, Yimeng and Zhang, Yang and Feng, Fuli and Lu, Jing and Zang, Xiaoxue and Lei, Chenyi and Song, Yang},
title = {GradCraft: Elevating Multi-task Recommendations through Holistic Gradient Crafting},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3637528.3671585},
doi = {10.1145/3637528.3671585},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {4774–4783},
numpages = {10},
keywords = {gradient crafting, multi-task learning, recommender system},
location = {Barcelona, Spain},
series = {KDD '24}
}
```
