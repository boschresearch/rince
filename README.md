# Ranking Info Noise Contrastive Estimation: Boosting Contrastive Learning via Ranked Positives

Official PyTorch implementation of the AAAI 2022 paper [Ranking Info Noise Contrastive Estimation: Boosting Contrastive Learning via Ranked Positives](https://arxiv.org/abs/2201.11736). The code allows the users to reproduce and extend the results reported in the study.

# Overview

This repository provides the code for RINCE, a contrastive loss that can exploit ranking information, such that same samples are very close and similar samples close in the embedding space (see Figure below).

![RINCE teaser](figures/RINCE_teaser.png)

## Setup

Clone this repository.

```buildoutcfg
git clone https://github.com/boschresearch/rince.git
cd rince
```

Install minimal set of requirements
```
conda env create --file requirements.yml python=3.7
conda activate rince
```

## Datasets

Cifar should download automatically on demand.
Download TinyImageNet and split both the training and val set into folders with inliers and outliers. The class names for both sets can be found in
```
./datasets/tiny_imagenet_inliers.txt
./datasets/tiny_imagenet_outliers.txt
```
create a symlink to your data directories in the folder ./data

## Training

### Rince

To train RINCE out-in on Cifar100 run

```
python main_supcon.py --batch_size 512 --num_workers 8 --print_freq 100 --data_folder ./data --dataset cifar100 --cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.6 --similarity_threshold 0.5 --n_sim_classes 5 --use_supercategories True --use_same_and_similar_class True --mixed_out_in_log True --exp_name test_RINCE_out_in
```

for RINCE in run:

```
python main_supcon.py --batch_size 512 --num_workers 8 --print_freq 100 --data_folder ./data --dataset cifar100 --cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.6 --similarity_threshold 0.5 --n_sim_classes 5 --use_supercategories True --use_same_and_similar_class True --exp_name test_RINCE_in
```

for RINCE out

```
python main_supcon.py --batch_size 512 --num_workers 8 --print_freq 100 --data_folder ./data --dataset cifar100 --cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.6 --similarity_threshold 0.5 --n_sim_classes 5 --use_supercategories True --use_same_and_similar_class True --do_sum_in_log False --exp_name test_RINCE_out
```

### SCL

SCL in
```
python main_supcon.py --batch_size 512 --num_workers 8 --print_freq 100 --data_folder ./data --dataset cifar100 --cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.1 --similarity_threshold 0.0 --one_loss_per_rank False --n_sim_classes 1 --do_sum_in_log True --exp_name test_SCL_in
```

SCL out
```
python main_supcon.py --batch_size 512 --num_workers 8 --print_freq 100 --data_folder ./data --dataset cifar100 --cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.1 --similarity_threshold 0.0 --one_loss_per_rank False --n_sim_classes 1 --do_sum_in_log False --exp_name test_SCL_out
```


### Cross-Entropy

```
python main_ce.py --batch_size 512 --num_workers 8 --print_freq 100 --data_folder ./data --dataset cifar100 --cosine --learning_rate 0.8 --exp_name cifar100_cross_entropy
```

## Testing the Model


Set ```--ckpt ``` to the last checkpoint of the model.

To evaluate RINCE and SCL run for linear probe, retrieval on cifar 100 labels, retrieval on cifar 100 coarse labels, and OOD on cifar10 and TinyImageNet the respective script:

```
python main_linear.py --batch_size 512 --learning_rate 5 --dataset cifar100 --data_folder ./data --model resnet50 --model_type contrastive --ckpt $path2chkpt/last.pth
python retrieval.py --dataset cifar100 --data_folder ./data --labelset fine --model resnet50 --model_type contrastive --ckpt $path2chkpt/last.pth
python retrieval.py --dataset cifar100 --data_folder ./data --labelset coarse --model resnet50 --model_type contrastive --ckpt $path2chkpt/last.pth
python out_of_dist_detection.py --dataset_outliers cifar10 --data_folder_outliers ./data --dataset cifar100 --data_folder ./data --size 32 --model resnet50 --model_type contrastive --ckpt $path2chkpt/last.pth
python out_of_dist_detection.py --dataset_outliers tiny_imagenet --data_folder_outliers ./data --dataset cifar100 --data_folder ./data --size 32 --model resnet50 --model_type contrastive --ckpt $path2chkpt/last.pth
```

To evaluate cross-entropy run:

```
python main_linear.py --batch_size 512 --learning_rate 5 --dataset cifar100 --data_folder ./data --model resnet50 --model_type cross_entropy --ckpt $path2chkpt/last.pth
python retrieval_old.py --dataset cifar100 --data_folder ./data --labelset fine --model resnet50 --model_type cross_entropy --ckpt $path2chkpt/last.pth
python retrieval_old.py --dataset cifar100 --data_folder ./data --labelset coarse --model resnet50 --model_type cross_entropy --ckpt $path2chkpt/last.pth
python out_of_dist_detection.py --dataset_outliers cifar10 --data_folder_outliers ./data --dataset cifar100 --data_folder ./data --size 32 --model resnet50 --model_type cross_entropy --ckpt $path2chkpt/last.pth
python out_of_dist_detection.py --dataset_outliers tiny_imagenet --data_folder_outliers ./data --dataset cifar100 --data_folder ./data --size 32 --model resnet50 --model_type cross_entropy --ckpt $path2chkpt/last.pth
```

## Training and Testing ImageNet-100

Check out the ImageNet branch of this repository.

## Pretrained Models

Pretrained models can be found in [trained_models](trained_models).

## Citation
If this code is useful in your research we would kindly ask you to cite our paper.
```
@article{rince2022AAAI,
    title={Ranking Info Noise Contrastive Estimation: Boosting Contrastive Learning via Ranked Positives},
    author={Hoffmann, David T and Behrmann, Nadine and Gall, Juergen and Brox, Thomas and Noroozi, Mehdi},
    journal={arXiv preprint arXiv:2201.11736},
    year={2022}
}
```

##  License

This project is open-sourced under the AGPL-3.0 license. See the [License](LICENSE) file for details.

For a list of other open source components included in this project, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the project
This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Contact
Please feel free to open an issue or contact us personally if you have questions, need help, or need explanations.
Write to one of the following email addresses, and maybe put the other in the cc:

david.hoffmann2@de.bosch.com

nadine.behrmann@de.bosch.com
