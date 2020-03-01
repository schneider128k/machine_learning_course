**Homework 3**

---

**Problem 1**

Implement the function ```get_random_data(w, b, mu, sigma, m)``` that generates random data for logisitic regression with two features features ```x_1``` and ```x_2```. This function should return the array ```data``` of shape ```(m, 2)``` and the array ```labels``` of shape ```(m, 1)```.

The entries of the arrays should be generated as follows.  For each row ```i in {0, 1, ..., m-1}```:

- Choose class label ```c=0``` with probability 1/2 and ```c=1``` with probability 1/2.  
- Choose the first feature ```x_1``` uniformly at random in the interval ```[0, 1)```. 
- Set the second feature ```x_2``` to be ```x_2 = w * x_1 + b + (-1)^c * n```, where the "noise" ```n``` is chosen according to the normal distribution with mean ```mu``` and standard deviation ```sigma```.
- The ith row of the array ```data``` consists of the features ```x_1``` and ```x_2```.
- The ith entry of the vector ```labels``` is the class label ```c```.

Implement the function ```display_random_data``` that takes as input the above two arrays ```labels``` and ```data```. It should create scatter plot of the 2D points stored in ```data```. Use red dots to plot the points whose labels are 1 and blue dots to plot the points whose labels are 0. 

Hints: You should see that the 2D points (feature vectors) corresponding to different classes are approximately separated by the line ```y = w * x + b```, where ```w``` and ```b``` are the parameters that you used to generate the data.  Note that the smaller the parameter ```mu```, the closer the points are to this line. Also, the larger the parameter ```sigma```, the more points can be on the wrong side of this line.

Experiment with different values of ```mu``` and ```sigma```.  Make sure that the parameter ```m``` is large enough so you have enough data points.

Split the data/labels into a training set (80%) and a test set (20%).

Links to the numpy documentation of the functions that can be used to draw samples according to the uniform and normal distributions:

- [Normal distribution](https://docs.scipy.org/doc/numpy-1.17.0/reference/random/generated/numpy.random.Generator.normal.html)
- [Uniform distribution](https://docs.scipy.org/doc/numpy-1.17.0/reference/random/generated/numpy.random.Generator.uniform.html)

You can learn more about the normal distribution on [https://en.wikipedia.org/wiki/Normal_distribution](https://en.wikipedia.org/wiki/Normal_distribution). To gain some intuition, it would be helpful to plot the Gaussian function for different parameters ```mu``` and ```sigma``` in a seperate notebook (that you do not have to submit).  Later in the semester, you will need to work with normal distribution to understand variational autoencoders.

---

**Problem 2**

Create a Keras to implement logistic regression with two features and train it with the data generated in Problem 1. The loss should be the binary cross entropy loss. 

How well does the trained model separate the red and blue dots?  You can obtain the separating line determined by the model by extracting the weights from the dense layer using the function ```get_weights```. See [https://keras.io/layers/about-keras-layers/](https://keras.io/layers/about-keras-layers/).  

Create a plot showing the random data, the true line used to generate the data, and the separating line of the trained model. Make sure that you describe in detail in your notebook how you proceed to obtain the separating line.

Note that you have to carry out some simple steps to obtain the separating line from the model weights (the two weights and the bias term of the dense layer).  This is not immediately obvious.  It maybe helpful to take a look at the heatmap below.

The trained model realizes function ```f : R^2 -> R``` that takes two features as input and outputs a number in the interval ```[0, 1]```. Use a heatmap to visualize this function.  

---

**Problem 3**

Use numpy to implement a logistic regression model from scratch and train it with the data generated as in Problem 1.

Hints: Look at the notes on logistic regression to figure out what the gradient is of the binary cross entropy loss with respect to ```w``` and ```b```. Note that you only have to implement stochastic gradient, that is, you do not have to write vectorized code for mini-batch gradient descent.

Create a plot showing the random data, the true line used to generate the data, and the separating line of the trained model.

Use a heatmap to visualize the function defined by your trained model.

You also have to compute the binary cross entropy loss and accuracy on the test set.

---

You have to post the links to your editable notebooks in Webcourses, save the notebooks in your GitHub course repo in the folder ```HW_3```.  Make sure that you follow the style guide for Python code.  Also, you should use text cells with markup headings to break up your notebooks into logical units. Always cite all sources used.
