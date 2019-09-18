**Homework 2**

**Problem 1**

Come up with handcrafted features for the MNIST digit classificiation problem and use them in a **linear** model with softmax output. That is, instead of feeding in the pixel values into the neural network, you feed in your handcrafted features.

The features could be: width and height of the digit, number of white regions (a typical 8 has three, a typical 6 has two components, and 2 has 1), average intensity, etc. 

The goal of this assignment is not to try to build the best possible classifier. The problem of computing the number of independent white regions is basic problem for job interviews. It boils down to computing connected components of a graph (which you studied in CS2).

**Problem 2**
