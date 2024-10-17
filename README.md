# Landmark-Based-Adaptive-Graph-Convolutional-Network for Facial Expression Recognitio
⭐️ Your star shines on us. Star us on GitHub!

⭐️ Our LBAGCN is published in https://ieeexplore.ieee.org/abstract/document/10683688
![image](https://github.com/user-attachments/assets/e29b22f9-a13b-48b7-8cfe-e818b579c2dc)


## The related software versions:
Python 3.9

* mmcv 1.5.0 
* torch 1.10.0 + cu113
* dlib 19.23.0

## Data preparation:
As show in file **data_process**
* ck+\\make_ck+_10_fold.py:
  
  used to build the training and testing datasets(10 fold) for CK+ dataset.
* oulu\\track_face.py:

  used to get the landmarks from the image in Oulu-CASIA dataset
* oulu\\make_oulu_10_fold.py
  
  used to build the training and testing datasets(10 fold) for Oulu-CASIA dataset.

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
* The format of the dataset is the same as it in [pyskl](https://github.com/kennymckormick/pyskl]).
  
* Some path may need to be modified to match your computer.

* All the codes are used in **windows**. So if you want to run it in Linux, there may have some changes.

## Acknowledgments
This project is based on the following codebase.

[pyskl](https://github.com/kennymckormick/pyskl])

