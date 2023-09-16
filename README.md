# RealHNS
The source code is for the paper: [Exploring False Hard Negative Sample in Cross-Domain Recommendation](https://dl.acm.org/doi/pdf/10.1145/3604915.3608791) accepted in Recsys 2023 by Haokai Ma, Ruobing Xie, Lei Meng, Xin Chen, Xu Zhang, Leyu Lin and Jie Zhou.

## Overview
This paper proposes a novel model-agnostic Real Hard Negative Sampling (RealHNS) framework specially for cross-domain recommendation (CDR), which aims to discover the false and refine the real from all HNS via both general and cross-domain real hard negative sample selectors. For the general part, we conduct the coarse- and fine-grained real HNS selectors sequentially, armed with a dynamic item-based FHNS filter to find high-quality HNS. For the cross-domain part, we further design a new cross-domain HNS for alleviating negative transfer in CDR and discover its corresponding FHNS via a dynamic user-based FHNS filter to keep its power.
![_](./framework.png)

## Dependencies
- Python 3.8.10
- PyTorch 1.12.0+cu102
- pytorch-lightning==1.6.5
- Torchvision==0.8.2
- Pandas==1.3.5
- Scipy==1.7.3

## Implementation of RealHNS
Delayed by other submission tasks, we are currently organizing the dataset and code and will make it public within a week.

