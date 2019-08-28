# HW 2

The goal of this homework is to create a convolutional neural network for the CIFAR10 data set. 
See [this colab notebook](https://colab.research.google.com/drive/1LZZviWOzvchcXRdZi2IBx3KOpQOzLalf) how to load the CIFAR data in Keras.

You should not use any pretrained convnets that come with Keras. You have to create and train your own convnets with Keras from scratch.

## Simple hold-out validation

Make sure that the data is divided into: 

- training set (80%)
- validation set (20%)
- test set. 

Use the training set to train your neural networks. Evaluate their performance on the validation data set. 

After trying several different architectures, choose the one that performs
best of the validation set. Try at least four different architectures by using data augmentation, using dropout, varying the number of layers, the number of filters, etc.

Train this final architecture on the data from the training set and validation set and evaluate its performance on 
the test set.

## k-fold validation

Reevaluate your best architecture using k-fold validation with k=5, that is, the size of the validation fold is 20%. Does the accuracy/loss obtain by k-fold validation differ from the accuracy/loss obtain by simple hold-out validation.

