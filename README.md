# cikm17-NNCF(Pytorch)

Implementation of Neighborhood-based Neural Collaborative Filtering model (NNCF) 

Ting Bai et al. "A Neural Collaborative Filtering Model with Interaction-based Neighborhood." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM, 2017.

## Run the model: python main.py


Parameters:

neigh_sample_num: the maximum neighbors in our algorithm

neg_num: the number of negative samples in training

embed_size: the  dimension of embedding

hidden_size: the dimension of MLP layer

epoch: training epoch

dropout: Parameters of the dropout function

lr: learning rate

l2: wight_decay

conv_kernel_size: the size of convolution

pool_kernel_size: the size of pooling

patience: early stopping 

## File Description

utils.py: Define data loading function, loss function, evaluation function

preprocess.py: Preprocess the original data set to generate  train.csv、dev.csv、test.csv

model.py: Define the model NNCF

neigh.py: Get neighbor information of a node (Louvain or Direct)

main.py: Entrance of the entire program

The python files are independent to make our project more flexible and extensible. You can tuning parameters and run the corresponding python file that you need.


## Requirement

Python version: 3.8.5

Pytorch version: 1.5.1

community: 0.14

networkx: 2.4


## Cite

Please cite our paper if you use this code in your own work:

@inproceedings{bai2017neural,<br>
  title={A neural collaborative filtering model with interaction-based neighborhood},<br>
  author={Bai, Ting and Wen, Ji-Rong and Zhang, Jun and Zhao, Wayne Xin},<br>
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},<br>
  pages={1979--1982},<br>
  year={2017},<br>
  organization={ACM}<br>
}
