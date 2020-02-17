# CAP 4630 Artificial Intelligence 

Undergraduate course on ML/AI at the University of Central Florida.


### Overview ###

- [Artificial intelligence, machine learning, and deep learning](https://github.com/schneider128k/machine_learning_course/blob/master/slides/1_a_slides.pdf)

- [Supervised learning, unsupervised learning, and reinforcement learning](https://github.com/schneider128k/machine_learning_course/blob/master/slides/1_b_slides.pdf)

---

### Fundamental machine learning concepts ###

- [Labeled/unlabeled examples, training, inference, classification, regression](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_a_slides.pdf)

- [Linear regression](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_b_slides.pdf)

- [Loss, empirical risk minimization, squared error loss, mean square error loss](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_c_slides.pdf)

- [Iterative approach to loss minimization](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_d_slides.pdf)

- [Gradient descent, learning rate](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_e_slides.pdf)

- [Stochastic gradient descent, mini-batch gradient descent](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_f_slides.pdf)

- [Anatomy of a neural network](https://github.com/schneider128k/machine_learning_course/blob/master/slides/anatomy_of_neural_network.md)

---

### Python, numpy, and matplotlib ###

- [Python](https://colab.research.google.com/drive/1CdwvzRhFNIlhp1mWVWFXSXAsmRFWvASq)

- [Style guide for Python code](https://www.python.org/dev/peps/pep-0008/)

- [numpy](https://colab.research.google.com/drive/1oixeI3yWIKkjCyg7SVIV-1mXI3LUbCAN)

- [matplotlib](https://colab.research.google.com/drive/1kKQ9sSFIrLlrZ0pW8nwjmLL9CUNUaSIC)


---

### Effect of learning rate on gradient descent for finding minima of univariate functions ###

Let's examine what could go wrong when applying gradient descent with a poorly chosen learning rate. We could fail to find any solution due to divergence or we could get stuck in a bad local minimum. The following notebook allows us to apply gradient descent for finding minima of univariate functions. (Univariate means that the functions depend on only one variable.)

- [Notebook for experimenting with different learning rates](https://colab.research.google.com/drive/1eECClMU1r-Y9hzPnRw89__jC3nw3C-zD)

---

### Visualization of bivariate functions ###
  
The loss function for a deep neural network depends on millions of parameters. Such functions are called multivariate because they depend on multiple variables.  It is no longer possible to easily visualize multivariate functions.

The following notebooks present two methods for visualizing bivariate function, that is, those that depend on exactly two variables.  Such functions define surfaces in 3D.  Think of the surface of a mountain range.

- [Notebook for creating density and contour plots](https://colab.research.google.com/drive/1pcvtvK6jITbp1Sf2nD2uEaDGpwUOA3IL)

- [Notebook for creating three dimensional plots](https://colab.research.google.com/drive/1btvbObh-nZ4MSC7QkjpS3RGpefN_msth)

---

### Linear regression using gradient descent - numpy implementation

- [Mathematical derivation of gradient for linear regression](https://github.com/schneider128k/machine_learning_course/blob/master/slides/linear_regression_simple.pdf)

In the first implementation, we consider the weight and bias separately and implement stochastic gradient descent.  It is easy to see the correspondance between the code and the mathematical expression for the gradient (see section 1 of the above notes).

- [Notebook for solving linear regression using stochastic gradient descent](https://colab.research.google.com/drive/1ZKa5sIiSgS8P1RuNyH6yYcZ6F9S7Yiwu)

In the second implementation, we combine the weight and bias into one vector. We also consider three versions of gradient descent: batch, mini-batch, and stochastic gradient descent.  We use a *vectorized* implementation, that is, all data in a batch is processed in parallel. It is more difficult to see the correspondance between the code and the mathematical expression for the gradient (see subsection 2.2 of the above notes).

 - [Notebook for solving linear regression using gradient descent (batch, mini-batch, and stochastic)](https://colab.research.google.com/drive/1qBxfTPoNcSFvpwu1NDl1V6cHEqL3aQl-)

This vectorized implementation of gradient descent for linear regression with a single feature can be generalized to linear regression with multiple features (you have to do this for n=2 for one of the homework problems).

---

### Linear regression using the normal equation - numpy implementation 

There is a closed-form solution for choosing the best weights and bias for linear regression. The optimal solution achieves the smallest squared error loss.  I will not cover this in class. If you are interested, you can find more details in the notes [Linear regression using the normal equation](https://github.com/schneider128k/machine_learning_course/blob/master/linear_regression_normal_equation.md).


---

### TensorFlow and Keras

Keras is a high-level deep learning API that allows you to easily build, train, evaluate, and execute all sorts of neural networks. Its documentation (or specification) is available at [https://keras.io](https://keras.io). The reference implementation [https://github.com/keras-team/keras](https://github.com/keras-team/keras) also called Keras, was developed by Francois Chollet as part of a research project and released as an open source project in March 2015. To perform the heavy computations required by neural networks, this reference implementation relies on a computation backend. At present, you can choose from three popular open source deep learning libraries: TensorFlow, Microsoft Cognitive Toolkit (CNTK), and Theano. Therefore, to avoid any confusion, we will refer to this reference implementation as *multibackend Keras*.

Since late 2016, other implementations have been released. You can now run Keras on Apache MXNet, Apple's Core ML, JavaScript or TypeScript (to run Keras code in a web browser), and PlaidML (which can run on all sorts of GPU devices, not just Nvidia).

TensorFlow 2 itself now comes bundled with its own Keras implementation, ```tf.keras```. It only supports TensorFlow as the backend, but it has the advantage of offering some very useful extra features: for example, it supports TensorFlow's Data API, which makes it easy to load and preprocess data efficiently.

---

### ```tf.keras``` 

In this course, we will use TensorFlow 2.x and ```tf.keras```.  Always make sure that you use correct versions of TensorFlow and Keras.

- [Notebook showing how to load TensorFlow 2](https://colab.research.google.com/notebooks/tensorflow_version.ipynb)

- [Notebook showing how to load ```tf.keras```](https://colab.research.google.com/drive/1fjMFLEJIXoC1LPCEXe4EUWCmRVfQVTu5)

- [TensorFlow 2](https://www.tensorflow.org/overview/)
  
- [keras.io](https://keras.io/) is the documentation for the multibackend Keras implementation.  You have to tweak the code examples from keras.io to use them with TensorFlow 2.x and ```tf.keras```.

---

### Linear regression - Keras implementation

Let's see how we can solve the simplest case of linear regression in Keras.

- [Notebook for solving linear regression](https://colab.research.google.com/drive/1pOFL4Qm6WOn2Nxxy6_HteEqQMxStTwzs)

---

### Keras datasets

We are going to work with some simple datasets to start learning about neural network. The collection ```tf.keras.datasets``` contains only a few simple datasets and provides an elementary way of loading them. (Later, we will learn about TensorFlow datasets, which contains 100 datasets and provides a high-performace input data pipelines to load the datasets.)

- [Notebook for loading and exploring the MNIST digits dataset](https://colab.research.google.com/drive/1HDZB0sEjhd0sdTFNCmJXvB8hYnE9KBM7)

- [Notebook for loading and exploring the CIFAR10 dataset](https://colab.research.google.com/drive/1LZZviWOzvchcXRdZi2IBx3KOpQOzLalf)

- [Notebook for loading and exploring the IMDB movie reviews dataset](https://colab.research.google.com/drive/1rYSzV6if2Y4P-X6yrF10IONLynyd416_)

---

### Keras basics
  
Let's briefly describe Keras concepts such as dense / convolutional / recurrent layers, sequential models, functional API, activation functions, loss functions, optimizers, and metrics.
  
- [Keras basics](https://github.com/schneider128k/machine_learning_course/blob/master/keras_basics.md)

---

### Keras models for classification of MNIST digits and fashion items

Before formally defining sequential neural networks with dense layers, let's look at some simple Keras models showing how to use such networks for classification. We consider the problems of classifying images from the MNIST digits dataset and the fashion items dataset. 
These problems are so-called *multi-class* / *single-label* classifications problems. 

*Multi-class* means that there are several classes. For instance, T-shirt, pullover or bag in the fashion items dataset. 

*Single-label* means that classes are mutually exclusive. For instance, an image is either the digit 0, or the digit 1, etc. in the MNIST digits dataset.

The example neural networks in the notebooks below consist of three layers: input, hidden, and output layers.
They use the *softmax* activation function in the last (output) layer and the *categorical cross entropy* loss function because the problems are multi-class, single-label classification problems.  They also use the  *relu activation* activation function for the hidden layer.  

These notebooks also show how to split datasets into *training datasets* and *test datasets* and also discuss *overfitting*.
    
- [Notebook for classifying MNIST digits with dense layers and analyzing model performance](https://colab.research.google.com/drive/144nj1SRtSjpIcKZgH6-GPdA9bWkg68nh)
  
- [Notebook for classifying fashion items with dense layers and analyzing model performance](https://colab.research.google.com/drive/1TTO7P5GTmsHhIt_YGqZYyw4KGBCnjqyW)

The notebook below uses ```pandas.DataFrame``` to display learning curves and to visually analyze predictions. 

- [Notebook for classifying fashion items with dense layers and analyzing model performance](https://colab.research.google.com/drive/1ejWWMNlsfnMMPlSNCAw6CykWH1Wlu_6D)

---

### Generalization, overfitting, and splitting dataset in train set and test set

The goal of machine learning is to obtain models that perform well on new unseen data.  It can happen that a model performs perfectly on the training data, but fails on new data.  This is called *overfitting*.  The following notes explain briefly how to deal with this important issue.

 - [Generalization, overfitting, and train & test sets](https://github.com/schneider128k/machine_learning_course/blob/master/slides/5_slides.pdf)


---

### Simple hold-out validation and K-fold validation

- [Simple hold-out validation and K-fold validation](https://github.com/schneider128k/machine_learning_course/blob/master/slides/6_slides.pdf)

---

### Binary classification, logistic regression, sigmoid activation, binary cross entropy loss

Logistic regression is used for binary classification problems. *Binary* means that there are only two classes.  For instance, a movie review has to be classified as either positive (class 1) or negative (class 0).  There is only one output neuron whose activation indicates the probability of class 1.  This output neuron uses the *sigmoid activation function*, which enforces that its activation inside the interval \[0, 1\], that is, is a valid probability. 

The *squared error loss* could be used, but it is much better to use the *binary cross entropy loss* instead of the *squared error loss* because it speeds up training.  The notes below derive the gradient for the two combinatations: sigmoid activation with squared error loss and sigmoid activation with binary cross entropy loss.

- [Logistic regression notes](https://github.com/schneider128k/machine_learning_course/blob/master/slides/logistic_regression.pdf)

The notebook below presents a simple elementary method for preprocessing text data so it can be input into a neural network.  We will discuss more advanced methods for preprocessing text later.

This notebook also shows how we can use a validation set to monitor the performance of the model and subsequently choose a good number of epochs to prevent overfitting.

- [Notebook for classifying IMDB movie reviews with dense layers](https://colab.research.google.com/drive/1e1scwvXdgCEEdPX4szVuw_KHD_NlINTZ)

---

### Multi-class / single-label classification, softmax activation, categorical cross entropy loss

We already talked briefly about multi-class / single-label classification, softmax activation, and categorical cross entropy loss when presenting Keras examples for classifying MNIST digits and fashion items.

The notes below explain the mathematics behind softmax activation and categorical cross entropy loss and derive the gradient for this combination of activation and loss.  

- [Softmax activation, categorical cross entropy](https://github.com/schneider128k/machine_learning_course/blob/master/slides/softmax.pdf)
  
- [Notebook for verifying formulas for the partial derivatives inside the gradient with symbolic differentiation](https://colab.research.google.com/drive/1G8u6w3FFhZyb0nWfparVvn77DSjHyxEW)

---

### Multi-class / multi-label classification

Image that you receive an image of a face and that you have to decide (a) if the person is smiling or not and (b) if the person is wearing glasses.  Similing and wearing glasses are independent of each other.  This is an example of multi-class / multi-label classification.

*Sigmoid activation* functions are used in the output layer in multi-class / multi-label classification problems. The number of output neurons is equal to the number of classes, and each neuron uses the sigmoid activation function.  The *binary cross entropy loss* is used for each output neuron.

We will look at some examples of multi-class / multi-label classification after introducting convolutional neural networks.  

---

### Two simple methods for fitting overfitting: dropout and L1 / L2 regularization

TO DO: create notebook

---

### Notes on backpropagation algorithm for computing gradients in sequential neural networks with dense layers

These notes explain how to compute the gradients for neural networks consisting of multiple dense layers.  I will not go over the mathematical derivation of the backpropagation algorithm.  Fortunately, the gradients are computed automatically in Keras.

My notes are mostly based on chapter 2 "How the backpropagation algorithm works" of the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).  

- [Notes on forward propagation, backpropagation algorithm for computing partial derivatives wrt weights and biases](https://github.com/schneider128k/machine_learning_course/blob/master/slides/neural_networks.pdf)

###  Numpy implementation of backpropagation algorithm 

My code is based on the code described in chapter 5 "Getting started with neural networks" of the book [Deep Learning and the Game of Go](https://www.manning.com/books/deep-learning-and-the-game-of-go).

- [Code for creating sequential neural networks with dense layers and training them with backprop and mini-batch SGD](https://github.com/schneider128k/machine_learning_course/blob/master/code/neural_network.py); currently, code is limited to (1) mean squared error loss and (2) sigmoid activations.

---

**TO DO: clean up everything below**

---

### Deep learning for computer vision (convolutional neural networks)

  - [CNN slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/CNN_slides.pdf)
  
  TO DO: add note of preventing overfitting with data augmentation (also, add L2/L1 regularization and dropout earlier!)

  **Classification of MNIST digits and fashion items**

  - [Colab notebook for classifying MNIST digits with convolutional layers and analyzing model performance](https://colab.research.google.com/drive/1HA-PoWkxgYa37McvUrA_n609jkKT5F7m)
  
  - [Colab notebook for classifying MNIST fashion items with convolutional layers and anlyzing model performance](https://colab.research.google.com/drive/1eI47U0vW_5ok6rVvNuenXS2EkTbqxcCV)
  
   **Transfer learning: classification of cats and dogs**
   
   based on Chapter 5 *Deep learning for computer vision* of the book *Deep learning with Python* by F. Chollet
   
   - [training convnet from scratch](https://colab.research.google.com/drive/1Jem-Kw-1z3TyFpgVfkxmfapPkH7wxh00)
   
   - [training convnet from scratch, using data augmentation and dropout](https://colab.research.google.com/drive/1bOov0VTMHNP_zasrSVPrDdzI3MBMcxgW)
   
   - [using VGG16 conv base for fast feature extraction (data augmentation not possible), using dropout](https://colab.research.google.com/drive/1Vame3KCI2KtnTLXnTXK0ZRaT8YjRErGz)
   
   - [using VGG16 conv base for feature extraction, using data augmentation, not using dropout](https://colab.research.google.com/drive/1pRyVUWTWGVBsJUYiR8rGgBAMva2ICdmi)
   
   - [using VGG16 conv base for feature extraction, using data augmentation, not using dropout, fine-tuning](https://colab.research.google.com/drive/1F-RWvoxH8MmT7c1UmNy41iuOp-ejiLoF)
  
  ---
  
  !!! Remove the notebooks below? Redundant ? !!!
  
 based on [Google ML Practicum: Image Classification](https://developers.google.com/machine-learning/practica/image-classification/)
    
  - [Colab notebook for training a convolutional neural network from scratch](https://colab.research.google.com/drive/1GCz7d32nfYTlY1paDk7-2oVw6E7HFK80)
   
  - [Colab notebook for training a CNN from scratch with data augmentation and dropout](https://colab.research.google.com/drive/1R67uzd5n_v2qnQh6klVCyBFCHTBRWZnF)
  
  - [Colab notebook for fine-tuning Google's Inception v3 model](https://colab.research.google.com/drive/1uVLIUWdT7--b59vM7NaSHkB-qFcu30jU)
  
 ---
  
  **Visualizing what convnets learn**
  
  based on chapter 5 *Deep learning for computer vision* of the book *Deep learning with Python* by F. Chollet
  
  - [Visualizing intermediate activations](https://colab.research.google.com/drive/12Y80BfKlSI8PU0-KNwcSvmxb6z2shZEl)
  
  - [Visualizing convnet filters](https://colab.research.google.com/drive/1Q8Iu_DvKnvvowNOuse0H-jl5gLFzTqaC), the convnet filter visualizations at the bottom of the notebook look pretty cool!
  
  - [Visualizing heatmaps of class activations](https://colab.research.google.com/drive/1Cg2Qy2JGc4XNJzkcaoDPc2hiFejGKdUd)
  
  - [Visualizing heatmaps of class activations, modified version](https://colab.research.google.com/drive/1KDdxUlvHsEAUmSHfiTh9kSkm9ceOz3tw), changes softmax to linear activation in last layer
  
  - [keras-vis](https://github.com/raghakot/keras-vis)
  This is a package for producing cool looking visualizations.  I had problems using it on colab.  !!! Fix it !!!
  
  ---
  
  **Some cool looking stuff**
  
  Based on Section 8.2 *DeepDream* and Section 8.3 *Neural style transfer* of the book *Deep learning with Python* by F. Chollet. I am not going to explain in detail how deep dream and neural style transfer work. I just wanted to include these notebooks to show you two cool examples of what can be done with deep neural networks.
  
  - [Deep dream](https://colab.research.google.com/drive/1AYaS92Da6xEPxQkMToAnqM5ThSzTf4E7)
  
  - [Neural style transfer](https://colab.research.google.com/drive/1OJGyFUsIImZ8nEN6A4zfaTXGzM6C_SIB)
  
 ---
 
### Deep learning for computer vision (residual networks)

The goal is to introduce more advanced architectures and concepts. This is based onthe Keras documentation: 
[CIFAR-10 ResNet](https://keras.io/examples/cifar10_resnet/).

The relevant research papers are:

- [Deep residual learning for image recognition](https://arxiv.org/pdf/1512.03385.pdf)

- [Identity mappings in deep residual networks](https://arxiv.org/pdf/1603.05027.pdf)

**Notebooks**

- [Resnet for CIFAR10 - train/val/test](https://colab.research.google.com/drive/13Pw71ozcGqF4QyRpTApe7J8sJS0JEQmt)

I have made several changes to the code from the Keras documentation. In the above notebook, I had to change the number of epochs and the learning rate schedule because the model is only trained on 40k and validated on 10k, whereas the model in the Keras documentation is trained on 50k and not validated at all. I wanted to have a situation that is similar to the situation in HW 2 so we can better compare the performance of the ResNet and the (normal) CNN.

- [Resnet for CIFAR10- train/test](https://colab.research.google.com/drive/1qpQc0senOmJZvVEBbOFcR8uoSP58L-5T)


---

### TensorFlow datasets 

TensorFlow datasets is a collection of nearly 100 ready-to-use datasets that can quickly help build high-performance input data pipelines for training TensorFlow models. Instead of downloading and manipulating datasets manually and then figuring out how to read their labels, TensorFlow datasets standardizes the data format so that it's easy to swap one dataset with another, often with just a single line of code change. As you will see later on, doing things like breaking the dataset down into training, validation, and testing is also a matter of a single line of code. The high-performance input data pipelines make it possible to work on the data in parallel. For instance, while the GPU is working with a batch of data, the CPU is prefeching the next batch.

- [TensorFlow datasets](https://www.tensorflow.org/datasets)

- [TensorFlow blog: introducing TensorFlow datasets](https://blog.tensorflow.org/2019/02/introducing-tensorflow-datasets.html)

- [Notebook for exploring MNIST dataset](https://colab.research.google.com/drive/1hpMxTkAjYn3qjLK5P_HiukQNrxgxacgu)
  
- [Notebook for exploring celebrities dataset ```celeb_a```](https://colab.research.google.com/drive/1jrabFFeiYU3ffXmnXh5szZud8A-zsV_5)

---

- one-shot learning
- image similarity, face-recognition

---

### Visualizing high-dimensional data using t-SNE

- [How to use t-SNE effectively?](https://distill.pub/2016/misread-tsne/)

- [PCA and t-SNE visualizations of MNIST fashion data set](https://colab.research.google.com/drive/1_PTPUXoWbtLhzluJeMT-eb_aPmC1uZlC)

- [t-SNE visualization of feature vectors for MNIST fashion data set](https://colab.research.google.com/drive/1WXIcE5ye6Vhm78RRDFfr9udu1bwijATl)
---
  
### Text
  
#### Character-based
  
- [Understanding LSTM networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [LSTM for text generation](https://colab.research.google.com/drive/1lo-zx2RVTFfaDaGu6R_VKnzgpZnzYAfi)

#### Word-based

- [Word embeddings](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- Using 1D convnets (TO D)
- [Word embeddings](https://colab.research.google.com/drive/1N3_B7Qz3Z5gC8uGHjoEYBhDGK1-EM-gC) (TO DO: change notebook !!!)
- Newsgroup classification with convolutional model using pretrained Glove embeddings (TO DO)
- IMDB sentiment classification with LSTM model (TO DO)
- ...

---
  
### One-shot learning ###

- [Exploring the omniglot dataset](https://colab.research.google.com/drive/1exJ-qNWj_m5Q-dcUesoL_fyBPyncfa0n)

- [One-shot learning for image recognition with the omniglot dataset](https://colab.research.google.com/drive/1q3vixnXSkolYz0kGv5nZiav6TRPTY5oJ)

---

### Variational Autoencoder

- [VAE for MNIST digits](https://colab.research.google.com/drive/1mU7A6OTm4N19Zc13ewzrpmIgMaOXPiyD)

---

### GANs

### Sequence-to-sequence models

- [Arguments ```return_sequences``` and and ```return_sequences```for LSTM cells in Keras](https://colab.research.google.com/drive/1xjndR7H9l6ICNBM0LqU-kfNheNt36tER)
- [Character-based sequence-to-sequence model for translating French to English](https://colab.research.google.com/drive/1WPRV12WdxXo7NzVqHwpqYzHAGdCPkmlx)
- TO DO: sequence-to-sequence model with attention

---

### Reinforcement learning ???

---

[Tools, additional materials](https://github.com/schneider128k/machine_learning_course/blob/master/tools_additional_materials.md)

