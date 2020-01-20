**Homework 2**

**Problem 1**

Come up with handcrafted features for the MNIST digit classificiation problem and use them in a simple model consisting of a single dense layer with softmax activation. 

Let x be the matrix representing a image of a MNIST digit.  Let vec(x) denote the flattened matrix x.

Your baseline model is the simple model that takes only vec(x) as input. The goal is to improve the accuracy of the simple model by augmenting the input vec(x) with hand-crafted features.  

Say, f1, f2, ..., fm are your hand-crafted features. The augmented input vector is then obtained by stacking vec(x) and the vector (f1, f2, ..., fm).  You can use the numpy command ```np.concatentate``` to stack vectors.

The features could be: 

- width of digit (normalized to be in the range (0, 1)
- height of digit (normalized to be in the range (0, 1)
- number of white regions (a typical 8 has three, a typical 6 has two components, and 2 has 1), average intensity, etc. 
- note the you can one-hot encode the number of connected components as follows: b1 = 1 iff num = 1, b2 = 1 if num = 2, b3 = 1 if num = 3, and b = 1 if num = 0 or num >= 4 (the latter case should normally not occur for well-formed digits); this is probably better than have just a single feature num / 3.

The problem of computing the number of independent white regions is a basic problem for job interviews. It boils down to computing connected components of the following graph. The vertices of the graph correspond to the pixels and are denoted by (i, j), which is the position of the pixel. Two vertices (i, j) and (i', j') are connected iff |i - i'| <= 1 and |j - j'| <= 1.  If a pixel at position (i, j) is black, then remove it together with its edges.

You have to compute the number of connected components for the 60 thousand images. So make sure that your code is efficient.

**Problem 2**

Modify the network architecture in [the notebook for classifying MNIST fashion items with dense layers and analyzing model performance](
https://colab.research.google.com/drive/1TTO7P5GTmsHhIt_YGqZYyw4KGBCnjqyW) by varying the number of hidden layers and choosing different sizes for the hidden layers.

Give three models consisting of only dense layers that 

- first model underfits (the model is too simple and cannot explain the data)
- second model overfits (the model is unnecessarily complex so it can easily adapt too much to the training data)
- third is pretty good (the model is either too simple, nor too complex; you don't train for too many epochs; you use dropout to fight overfitting)

Make sure that you plot the curves depicting the training/validation accuracy/loss.

**Problem 3**

In this problem you have to work with the CIFAR10 data set. Check out the notebook [cifar10_data_set](https://colab.research.google.com/drive/1LZZviWOzvchcXRdZi2IBx3KOpQOzLalf) to see how to load it.

Give three convolutional models that 

- first model underfits
- second model overfits
- third model is pretty good

Make sure that you plot the curves depicting the training/validation accuracy/loss.
