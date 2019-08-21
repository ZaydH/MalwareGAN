# Adversarial Malware Generation Using GANs

[![docs](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ZaydH/MalwareGAN/blob/master/LICENSE)

Implementation of a Generative Adversarial Network (GAN) that can create adversarial malware examples.  The work is inspired by **MalGAN** in the paper "[*Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN*](https://arxiv.org/abs/1702.05983)" by Weiwei Hu and Ying Tan.

Framework written in [PyTorch](https://pytorch.org/) and supports CUDA.

## Running the Script

The malware GAN is provided as a package in the folder `malgan`.  A driver script is provided in `main.py`, which processes input arguments via `argparse`.  The basic interface is:

    python main.py Z BATCH_SIZE NUM_EPOCHS MALWARE_FILE BENIGN_FILE

* `Z` -- Dimension of the latent vector.  Must be a positive integer.
* `BATCH_SIZE` -- Batch size for *malicious* examples.  The benign batch size is proportional to `BATCH_SIZE` and the fraction of total training samples that are benign.
* `NUM_EPOCHS` -- Maximum number of training epochs
* `MALWARE_FILE` -- Path to a serialized `numpy` or `torch` matrix where the rows represent a single **malware** file's binary feature vector.
* `BENIGN_FILE` -- Path to a serialized `numpy` or `torch` matrix where the rows represent a single **benign** file's binary feature vector.

For checkout purposes, we recommend calling:

    python main.py 10 32 100 data/trial_mal.npy data/trial_ben.npy 

## Dataset

A trial dataset is included with this implementation in the `data` folder.  The data was publish in the repository: [yanminglai/Malware-GAN](https://github.com/yanminglai/Malware-GAN).  This dataset should only be used for proof of concept and initial trials. 

We recommend the SLEIPNIR dataset.  It was published by ad-Dujaili et al.  The authors requested that the dataset not be shared publicly, and we respect that request.  However, researchers and students may request access directly from the authors as described on their [Github repository](https://github.com/ALFA-group/robust-adv-malware-detection).  Look for the link to the Google form.

## CUDA Support

The implementation supports both CPU and CUDA (i.e., GPU) execution.  If CUDA is detected on the system, the implementation defaults to CUDA support.

## Requirements

This program was tested with Python 3.6.5 on MacOS and on Debian Linux.  `requirements.txt` enumerates the exact packages used. A summary of the key requirements is below: 

* PyTorch (`torch`) -- Ver. 1.2.0
* Scikit-Learn (`sklearn`) -- Ver. 0.20.2
* NumPy (`numpy`)
* TensorboardX -- If runtime profiling is not required, this can be removed.
