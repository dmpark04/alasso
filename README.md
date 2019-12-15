# Continual learning by asymmetric loss approximation with single-side overestimation

This repository contains code to reproduce the key findings of:

Park, D., Hong, S., Han, B., & Lee, K. M. (2019). Continual learning by asymmetric loss approximation 
with single-side overestimation. 
In Proceedings of the IEEE International Conference on Computer Vision (pp. 3335-3344).

http://openaccess.thecvf.com/content_ICCV_2019/html/Park_Continual_Learning_by_Asymmetric_Loss_Approximation_With_Single-Side_Overestimation_ICCV_2019_paper.html

## BibTeX
```
@inproceedings{park2019continual,
  title={Continual learning by asymmetric loss approximation with single-side overestimation},
  author={Park, Dongmin and Hong, Seokil and Han, Bohyung and Lee, Kyoung Mu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3335--3344},
  year={2019}
}
```


## Requirements

We have tested this code with the following configuration:

* Python 3.6.9
* Tensorflow 1.13.1
* Keras 2.0.5

## How to run

For 30 tasks, run the following commands. 

```
cd permuted_minst
python train.py
python 'Basic graph Permuted MNIST.py'
```

For 100 tasks, run the following commands. 

```
cd permuted_minst
python train_100.py
python 'Basic graph Permuted MNIST.py'
```

