# HW 3

The goal of this homework is to learn how to leverage pretrained convnets and to use some visualization techniques.
You will work with the data set *cats vs dogs* and use the pretrained convnet VGG19.

Experiment with different classifiers, trying to maximize the validation accuracy. You only need to show one classifier.

---

## Problem 1

Do feature extraction with data augmentation.

## Problem 2

Do fine-tuning with data augmentation. 

## Problem 3

Visualize heatmaps of class activation for the the model obtained in Problem 2.

## Problem 4

Build an activation model that takes as input an image and produces as output the activation of the last conv layer of the model obtained in Problem 2. Using this activation model obtain the corresponding activations for the validation images. Apply t-SNE visualization to these activations to see how well the convnet separates cats from dogs.
