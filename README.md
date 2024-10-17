# Landmark-Based-Adaptive-Graph-Convolutional-Network for Facial Expression Recognitio
⭐️ Our LBAGCN is published in https://ieeexplore.ieee.org/abstract/document/10683688

⭐️ Your star shines on us. Star us on GitHub!

## The related software versions:
Python 3.9

* mmcv 1.5.0 
* torch 1.10.0 + cu113 

## Data:
As show in file **data process**
* xxx:
  
  used to process the CK+ dataset.
* yyy:
  
  used to process the Oulu-CASIA dataset.

## How to train:
As show in file **tools**

(there use the Oulu-CASIA dataset as an example)
* my_train_10_fold.py:

  used to train the single-stream model.
* my_train_multiflow.py:
  
  used to train the multi-stream model.
* test.py:
  
  used to test the given model.

## NOTE：
* Some path may need to be modified to match your computer.

* All the codes are used in **windows**. So if you want to run it in Linux, there may have some changes.

## Acknowledgments
This project is based on the following codebase.

[pyskl](https://github.com/kennymckormick/pyskl])

