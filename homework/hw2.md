**Homework 2**

**Problem 1**

Come up with handcrafted features for the MNIST digit classificiation problem and use them in a **linear** model with softmax output. 

Let *x* be the matrix representing a image of a MNIST digit.  Let vec(*x*) denote the flattened matrix *x*.

That is, instead of feeding in the pixel values into the neural network, you feed in your handcrafted features.

The features could be: width and height of the digit, number of white regions (a typical 8 has three, a typical 6 has two components, and 2 has 1), average intensity, etc. 

The goal of this assignment is not to try to build the best possible classifier. The problem of computing the number of independent white regions is basic problem for job interviews. It boils down to computing connected components of a graph (which you studied in CS2).

**Problem 2**

Modify the network architecture in [the notebook for classifying MNIST fashion items with dense layers and analyzing model performance](
https://colab.research.google.com/drive/1TTO7P5GTmsHhIt_YGqZYyw4KGBCnjqyW) by varying the number of hidden layers and choosing different sizes for the hidden layers.

Give an example of an architecture that 

- underfits
- overfits
- is pretty good (note that you cannot get perfect results with only dense layers)

Extra work: Read about dropout. Use it to decrease overfitting.


