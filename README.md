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

Download ImageNet, create the different ImageNet-100 splits with symbolic links using the following lists. You need to create both, a train and a val folder
Download AwA2.
```
./datasets/ImageNet-100.txt
./datasets/imagenet_outliers_split1.txt
./datasets/imagenet_outliers_split2.txt
./datasets/imagenet_outliers_split3.txt
```
create a symlink to your data directories in the folder ./data

## Training on ImageNet-100

### Rince 

To train RINCE out-in on ImageNet-100

```
python main_supcon.py --roberta_float_threshold=0.4 --mixed_out_in=True --do_sum_in_log=True --dataset=imagenet --similarity_ranking_imagenet=roberta_threshold --save_freq=100 --epochs=500 --min_tau=0.1 --max_tau=0.6 --use_dynamic_tau=True --memorybank_size=8192 --learning_rate=0.3 --batch_size=768 --cosine --size=224 --data_folder=./data/imagenet100/train --model=resnet18 --exp_name=rince_imagenet100 --dist-backend=NCCL --num_workers=16 --loss_type=rince
```


### SCL

```
python main_supcon.py --data_folder=./data/imagenet100/train --dataset=imagenet --save_freq=100 --cosine --batch_size=768 --size=224 --learning_rate=0.325 --min_tau=0.1 --max_tau=0.1 --memorybank_size=8192 --similarity_threshold=0.5 --do_sum_in_log=False --exp_name=imagenet100_SCL_out --model=resnet18 --num_workers=16 --dist-backend=NCCL --epochs=500 --similarity_ranking_imagenet=roberta_threshold --loss_type=supcon
```


### Cross-Entropy

```
python main_ce.py --data_folder=./data/imagenet100/train --dataset=imagenet_100 --cosine --model=resnet18 --learning_rate=0.3 --batch_size=768 --size=224 --exp_name=imagenet100_ce_baseline_sslAugment --epochs=500 --size=224 --num_workers=16 --save_freq=100 --use_ssl_augmentations=True --imagenet100=True
```

## Testing the Model


Set ```--ckpt ``` to the last checkpoint of the model. 

To evaluate RINCE and SCL run for linear probe, retrieval on cifar 100 labels, retrieval on cifar 100 coarse labels, and OOD on cifar10 and TinyImageNet the respective script:

To evaluate the models checkout the cifar branch and run the following:

```
python main_linear.py --model=resnet18_standard --batch_size 512 --learning_rate 5 --ckpt $path2chkpt/last.pth --data_folder=./data/imagenet100/images --dataset=imagenet_100 --size=224 --num_workers=32 --epochs=100
python out_of_dist_detection.py --model=resnet18_standard --dataset=imagenet_100 --dataset_outliers=AwA2 --data_folder=./data/imagenet100/images --data_folder_outliers=./data/Animals_with_Attributes2/JPEGImages --size=224 --ckpt $path2chkpt/last.pth --num_workers=8
python out_of_dist_detection.py --model=resnet18_standard --dataset=imagenet_100 --dataset_outliers=imagenet --data_folder=./data/imagenet100/images --data_folder_outliers=./data/imagenet_outliers_split1 --size=224 --ckpt $path2chkpt/last.pth --num_workers=8
python out_of_dist_detection.py --model=resnet18_standard --dataset=imagenet_100 --dataset_outliers=imagenet --data_folder=./data/imagenet100/images --data_folder_outliers=./data/imagenet_outliers_split2 --size=224 --ckpt $path2chkpt/last.pth --num_workers=8
python out_of_dist_detection.py --model=resnet18_standard --dataset=imagenet_100 --dataset_outliers=imagenet --data_folder=./data/imagenet100/images --data_folder_outliers=./data/imagenet_outliers_split3 --size=224 --ckpt $path2chkpt/last.pth --num_workers=8
```

## Pretrained Models

Pretrained models are coming soon. 

## Citation
If this code useful in your research we would kindly ask you to cite our paper.
```
@article{rince2022AAAI,
    title={Ranking Info Noise Contrastive Estimation: Boosting Contrastive Learning via Ranked Positives},
    author={Hoffmann, David T and Behrmann, Nadine and Gall, Juergen and Brox, Thomas and Noroozi, Mehdi},
    journal={arXiv preprint arXiv:2201:11736},
    year={2022}
}
```

##  License 

This project is open-sourced under the AGPL-3.0 license. See the [License](LICENSE) file for details.

For a list of other open source components included i nthis project, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the project 
This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Contact
Please feel free to open an issue or contact us personally if you have questions, need help, or need explanations.
Write to one of the following email addresses, and maybe put one other in the cc:

david.hoffmann2@de.bosch.com

nadine.behrmann@de.bosch.com
