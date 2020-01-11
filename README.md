# CAP 4630 Artificial Intelligence 

Undergraduate course on ML/AI at the University of Central Florida.

### Artificial intelligence, machine learning, and deep learning

- [1_a_slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/1_a_slides.pdf)

### Supervised learning, unsupervised learning, and reinforcement learning

- [1_b_slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/1_b_slides.pdf)

### Labeled/unlabeled examples, training, inference, classification, regression ###

- [2_a_slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_a_slides.pdf)

### Linear regression ###

- [2_b_slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_b_slides.pdf)

### Loss, empirical risk minimization, squared error loss, mean square error loss ### 

- [2_c_slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/2_b_slides.pdf)

### Gradient descent, stochastic gradient descent ###

- *working on slides*

- [Anatomy of a neural network](https://github.com/schneider128k/machine_learning_course/blob/master/slides/anatomy_of_neural_network.md)

### numpy and matplotlib

- [numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html)

- [matplotlib](https://matplotlib.org/tutorials/index.html#introductory)

### Effect of learning rate on gradient descent

Let's look what could go wrong when applying gradient descent. We could fail to find any solution due to divergence or we could get stuck in a bad local minimum. The following notebook allows us to look at functions that maps real numbers to real numbers.

  - [Notebook for experimenting with different learning rates](https://colab.research.google.com/drive/1eECClMU1r-Y9hzPnRw89__jC3nw3C-zD)
  
The loss function for a deep neural network depends on millions of parameters, that is, maps a high-dimensional real vectors to real numbers. It is no longer possible to visualize the shape of the function.  The following notebooks show you how to visualize functions that map two-dimensional real vectors to real numbers.

- [Notebook for creating density and contour plots](https://colab.research.google.com/drive/1pcvtvK6jITbp1Sf2nD2uEaDGpwUOA3IL)

- [Notebook for creating three dimensional plots](https://colab.research.google.com/drive/1btvbObh-nZ4MSC7QkjpS3RGpefN_msth)

### Linear regression using gradient descent - numpy implementation

To get started, let's consider the simple case of linear regression: n=1, that is, there is only one feature and one weight.

- [Mathematical derivation of gradient for linear regression](https://github.com/schneider128k/machine_learning_course/blob/master/slides/linear_regression_simple.pdf)

In the first implementation, we consider the weight and bias separately.

- [Notebook for solving linear regression using stochastic gradient descent](https://colab.research.google.com/drive/1ZKa5sIiSgS8P1RuNyH6yYcZ6F9S7Yiwu)

In the second implementation, we combine the weight and bias into one vector. We also consider three versions of gradient descent: batch, stochastic, and mini-batch gradient descent.

 - [Notebook for solving linear regression using gradient descent (batch, stochastic, mini-batch)](https://colab.research.google.com/drive/1qBxfTPoNcSFvpwu1NDl1V6cHEqL3aQl-)

### Linear regression - Keras implementation

Let's see how we can solve the simplest case of linear regression in Keras.

  - [Notebook for solving linear regression](https://colab.research.google.com/drive/1pOFL4Qm6WOn2Nxxy6_HteEqQMxStTwzs)

### Linear regression using the normal equation - numpy implementation 

 There is a closed-form solution for choosing the best weights and bias for linear regression. The optimal solution achieves the smallest squared error loss.
 
  - [Colab notebook for solving linear regression using normal equation](https://colab.research.google.com/drive/1J7yct9aGfhtfXw8n00Mq4R-xldSSM1WY)
 
  To understand the mathematics underlying the normal equation, read the following materials. I will not cover the derivation of the normal equation.

  - [Chapter 2 Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)
  
  - [Chapter 4 Numerical Computation, Section 4.3 Gradient-Based Optimization](https://www.deeplearningbook.org/contents/numerical.html) 
  
  - [Chapter 5 Machine Learning Basics, Subsection 5.1.4 Example: Linear Regression](https://www.deeplearningbook.org/contents/ml.html)
  
  - [Additional materials: proof of convexity of MSE and computation of gradient of MSE](https://github.com/schneider128k/machine_learning_course/blob/master/slides/linear_regression.pdf)

### TensorFlow and Keras

We will use Keras to build (almost) all deep learning models. Roughtly speaking, TensorFlow is a back-end for deep learning, whereas Keras is a front-front.  Keras can use TensorFlow or other backends.  

Keras is now part of the latest version of TensorFlow 2.0 so it is available automatically when you import tensorflow.  Previously (TensorFlow 1.x) you had to import Keras seperately.  I may need to do some minor tweaks to the notebooks so that everything is perfectly adapted to TensorFlow 2.0.

  - [TensorFlow](https://www.tensorflow.org/tutorials/)
  
  - [Keras](https://keras.io/)

  - [3 Slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/3_slides.pdf)

  - [4 Slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/4_slides.pdf)

### Keras examples

Let's now see how we can solve more interesting problems with Keras.  We consider the problems of classifying images from the MNIST digits, MNIST fashion items, and CIFAR10 datasets.  

The classifications problems are all multi-class, single-label classifications problems.  *Multi-class* means that there are several classes. For instance, T-shirt, pullover or bag in the MNIST fashion items dataset.  *Single-label* means that classes are mutually exclusive. For instance, an image is either the digit 0, or the digit 1, etc. in the MNIST digits dataset.

In the multi-class, single-label classification problem, the activation in the last layer is *softmax* and the loss function is *categorical cross entropy*.  

The examples below use the so-called *relu activation* function for the hidden layer.
  
  - [Notebook for loading and exploring the MNIST digits data set](https://colab.research.google.com/drive/1HDZB0sEjhd0sdTFNCmJXvB8hYnE9KBM7)
  
  - [Notebook for classifying MNIST digits with dense layers and analyzing model performance](https://colab.research.google.com/drive/144nj1SRtSjpIcKZgH6-GPdA9bWkg68nh)
  
  - [Notebook for classifying MNIST fashion items with dense layers and analyzing model performance](https://colab.research.google.com/drive/1TTO7P5GTmsHhIt_YGqZYyw4KGBCnjqyW)

  - [Notebook for displaying CIFAR10 data set](https://colab.research.google.com/drive/1LZZviWOzvchcXRdZi2IBx3KOpQOzLalf)

### Generalization, overfitting, splitting data in train & test sets

The goal of machine learning is to obtain models that perform well on new unseen data, that is.  For instance, it can happen that a model performs perfectly on the training data, but fails on new data.  This is called *overfitting*.  The following notes explain briefly how to deal with this important issue.

  - [5 Slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/5_slides.pdf)
  
### Validation

  - [6 Slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/6_slides.pdf)

### Logistic regression, gradient for squared error loss, and gradient for binary cross entropy loss

Logistic regression is used for binary classification problems. *Binary* means that there are only two classes.  For instance, an image has to be classified as either a cat or a dog.  There is only one output neuron whose activation indicates the class (say, 1=dog, 0=cat).  It is best to use the *binary cross entropy loss* instead of the *squared error loss*. 

*Sigmoid activation* functions are used in multi-class, multi-label classification problems. The number of output neurons is equal to the number of classes, and each neuron uses the sigmoid activation function.  The binary cross entropy loss is used for each output neuron.

  - [Logistic regression notes](https://github.com/schneider128k/machine_learning_course/blob/master/slides/logistic_regression.pdf)

### Softmax, gradient for categorical cross entropy loss

  - [Softmax, categorical cross entropy](https://github.com/schneider128k/machine_learning_course/blob/master/slides/softmax.pdf)
  
  - [Notebook for verifying formulas for partial derivatives with symbolic differentiation](https://colab.research.google.com/drive/1G8u6w3FFhZyb0nWfparVvn77DSjHyxEW)

### Sequential neural networks with dense layers

These notes explain how to compute the gradients for neural networks consisting of multiple dense layers.  I will not go over the mathematical derivation of the backpropagation algorithm.  Fortunately, the gradients are computed automatically in Keras.

  - [Notes on forward propagation, backpropagation algorithm for computing partial derivatives wrt weights and biases](https://github.com/schneider128k/machine_learning_course/blob/master/slides/neural_networks.pdf)
  
  - [Code for creating sequential neural networks with dense layers and training them with backprop and mini-batch SGD](https://github.com/schneider128k/machine_learning_course/blob/master/code/neural_network.py); currently, code is limited to (1) mean squared error loss and (2) sigmoid activations.

### Deep learning for computer vision (convolutional neural networks)

  - [CNN slides](https://github.com/schneider128k/machine_learning_course/blob/master/slides/CNN_slides.pdf)
  
  TO DO: add note of preventing overfitting with data augmentation (also, add L2/L1 regularization and dropout earlier!)

  **Classification of MNIST digits and fashion items**

  - [Colab notebook for classifying MNIST digits with convolutional layers and analyzing model performance](https://colab.research.google.com/drive/1HA-PoWkxgYa37McvUrA_n609jkKT5F7m)
  
  - [Colab notebook for classifying MNIST fashion items with convolutional layers and anlyzing model performance](https://colab.research.google.com/drive/1eI47U0vW_5ok6rVvNuenXS2EkTbqxcCV)
  
   **Classification of cats and dogs**
   
   based on Chapter 5 *Deep learning for computer vision* of the book *Deep learning with Python* by F. Chollet
   
   - [training convnet from scratch](https://colab.research.google.com/drive/1Jem-Kw-1z3TyFpgVfkxmfapPkH7wxh00)
   
   - [training convnet from scratch, using data augmentation and dropout](https://colab.research.google.com/drive/1bOov0VTMHNP_zasrSVPrDdzI3MBMcxgW)
   
   - [using VGG16 conv base for fast feature extraction (data augmentation not possible), using dropout](https://colab.research.google.com/drive/1Vame3KCI2KtnTLXnTXK0ZRaT8YjRErGz)
   
   - [using VGG16 conv base for feature extraction, using data augmentation, not using dropout](https://colab.research.google.com/drive/1pRyVUWTWGVBsJUYiR8rGgBAMva2ICdmi)
   
   - [using VGG16 conv base for feature extraction, using data augmentation, not using dropout, fine-tuning](https://colab.research.google.com/drive/1F-RWvoxH8MmT7c1UmNy41iuOp-ejiLoF)
  
  ---
  
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
  This is a package for producing cool looking visualizations.  I had problems using it on colab.  
  
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

### Sequence-to-sequence models

- [Arguments ```return_sequences``` and and ```return_sequences```for LSTM cells in Keras](https://colab.research.google.com/drive/1xjndR7H9l6ICNBM0LqU-kfNheNt36tER)
- [Character-based sequence-to-sequence model for translating French to English](https://colab.research.google.com/drive/1WPRV12WdxXo7NzVqHwpqYzHAGdCPkmlx)
- TO DO: sequence-to-sequence model with attention

---

[Tools, additional materials](https://github.com/schneider128k/machine_learning_course/blob/master/tools_additional_materials.md)

