**Homework 1**

**Problem 1** 

What does an average MNIST digit look like? For each i=0,1,...,9, compute the average of digit i and display it.

**Problem 2**

Come up with handcrafted features for the MNIST digit classificiation problem and use them in a **linear** model with softmax output. That is, instead of feeding in the pixel into the neural network, you feed in the handcraft features.

The features could be: width and height of the digit, number of white regions (a typical 8 has three, a typical 6 has two components, and 2 has 1), average intensity, etc. 

Look at under [Keras example in the README file](https://github.com/schneider128k/machine_learning_course/blob/master/README.md). There is a notebook showing how to load the MNIST data set etc.
