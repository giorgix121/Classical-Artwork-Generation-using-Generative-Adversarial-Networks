# Classical Artwork Generation using Generative Adversarial Networks

This repository contains the implementation and report for the project "Classical Artwork Generation using Generative Adversarial Networks" developed as a part of the CS-GY 6953 / ECE-GY 7123 Deep Learning course at New York University Tandon School of Engineering.

## Authors

- Giorgi Merabishvili (Department of Electrical and Computer Engineering, NYU)
- Aaftab Mohammad (Department of Electrical and Computer Engineering, NYU)
- Mohd Faizaan Khan (Department of Computer Science and Engineering, NYU)

## Project Overview

This project focuses on generating high-quality artwork images using a Generative Adversarial Network (GAN). The model is trained on the "Best Artworks of All Time" dataset sourced from Kaggle, which includes 8683 images of renowned paintings from various artists and art movements.

## Architecture

### Generator
The generator is designed with a two-stage approach:
- **Initial layers:** Focus on broader features using ConvTranspose2d layers.
- **Detail layers:** Refine these features using additional ConvTranspose2d layers and Tanh activation for the output.

### Discriminator
The discriminator employs:
- **Spectral normalized Conv2d layers** to ensure stable training.
- **LeakyReLU activations** and **dropout layers** for regularization.
- **Gaussian noise** added to input images to increase robustness.

## Methodology

### Loss Function
- **Binary Cross-Entropy (BCE) loss** for both generator and discriminator.
- Discriminator maximizes the log-likelihood of correctly classifying real and fake images.
- Generator minimizes the log-likelihood of the discriminator correctly identifying its outputs as fake.

### Training Procedure
- **Alternating updates** between the generator and discriminator.
- **Data augmentation** techniques such as random horizontal flips and rotations.
- **Dynamic learning rate adjustment** based on discriminator performance feedback.

### Hyperparameters
- Learning rates for the generator's initial and detail layers: 0.0002 and 0.00005.
- **Adam optimizer** with beta values of 0.5 and 0.999.
- Batch size of 64 and a total of 500 epochs.
- Latent vector size for the generator's input: 128.

## Results
The results demonstrate significant improvements in the quality and diversity of the generated images over the training period. The model was trained for 500 epochs, and the adaptive learning rate mechanism effectively balanced the training dynamics between the generator and discriminator, preventing issues such as mode collapse.

## Repository Contents

- `report/` - Contains the final project report in AAAI style.
- `src/` - Source code for the GAN model, data processing, and training scripts.
- `images/` - Sample images generated at different epochs during training.
- `models/` - Saved models at specified intervals for future fine-tuning and evaluation.
 
