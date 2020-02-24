**Homework 3**

**Problem 1**

Implement the function ```get_random_data(w, b, mu, sigma, m)``` that generates random data for logisitic regression with a single feature. This function should return the array ```data``` of shape ```(m, 2)``` and the array ```labels``` of shape ```(m, 1)```.

The entries of the arrays should be generated as follows.  For each row ```i in {0, 1, ..., m-1}```:

- Choose ```c=0``` with probability 1/2 and ```c=1``` with probability 1/2.  
- Choose ```x``` uniformly at random in the interval \[0, 1). 
- Set ```y = w * x + b + (-1)^c * n``` where ```n``` is chosen according to the normal distribution with mean ```mu``` and standard deviation ```sigma```.
- The ith row of the array ```data``` consists of the entries ```x``` and ```y```.
- The ith entry of the vector ```labels``` is ```c```.

Implement the function ```display_random_data``` that takes a input the above two arrays. It should create scatter plot of the 2D poins stored in ```data```. Use red dots to plot the points whose label are 1 and blue dots to plot the points whose labels are 0. Also, plot the line ```y = w * x + b```. 

Experiment with different values of ```mu``` and ```sigma```.

**Problem 2**

Use Keras to implement logistic regression. The loss should be the binary cross entropy loss. How well does the trained model separate the red and blue dots?  You can obtain the seperating line by extracting the weights from the dense layer using the function ```get_weights```. See [https://keras.io/layers/about-keras-layers/](https://keras.io/layers/about-keras-layers/).  Create a plot showing the random data, the line used to generate the data, and the separating line of the trained model.

**Problem 3**

Use numpy to implement logistic regression from scratch. Look at the notes on logistic regression to figure out what the gradient is of the binary cross entropy loss with respect to ```w``` and ```b```. Note that you only have to implement stochastic gradient, that is, you do not have to write vectorized code for mini-batch gradient descent.
