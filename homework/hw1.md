**Homework 1**

**Problem 1** 

What does an average MNIST digit look like? For each i=0,1,...,9, compute the average of digit i and display it.

**Problem 2**

Come up with handcrafted features for the MNIST digit classificiation problem and use them in a **linear** model with softmax output. That is, instead of feeding in the pixel values into the neural network, you feed in your handcrafted features.

The features could be: width and height of the digit, number of white regions (a typical 8 has three, a typical 6 has two components, and 2 has 1), average intensity, etc. 

Look  under [Keras examples in the README file](https://github.com/schneider128k/machine_learning_course/blob/master/README.md). There is a notebook showing how to load the MNIST data set etc.

The goal of this assignment is not to try to build the best possible classifier, just to get your started with Python. The problem of computing the number of independent white regions is basic problem for job interviews. It boils down to computing connected components of a graph (which you studied in CS2).
