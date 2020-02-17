**Homework 2**

**Problem 1**

Find three simple bivariate functions f1, f2, and f3 and three points p1, p2, p3 such that

- p1 is a minimum of f1
- p2 is a maximum of f2
- p3 is neither a minimum nor maximum of f3, but the gradient of f3 at p3 is the zero vector.

Use (a) three-dimensional plots as in [```three_dimensional_plotting.ipynb```](https://colab.research.google.com/drive/1btvbObh-nZ4MSC7QkjpS3RGpefN_msth) and
(b) density plots and (c) contour plots as in [```density_and_contour_plots.ipynb```](https://colab.research.google.com/drive/1pcvtvK6jITbp1Sf2nD2uEaDGpwUOA3IL). Indicate clearly the points p1, p2, and p3 in all plots.

**Problem 2**

Extend the code for mini-batch gradient descent in the notebook [```linear_regression_gradient_descent.ipynb```](https://colab.research.google.com/drive/1qBxfTPoNcSFvpwu1NDl1V6cHEqL3aQl-) 
to the case n=2, that is, the model parameters are the two weights w1 and w2 and the bias term b.  (You can remove the code for batch-gradient descent and stochastic gradient descent.) Make sure that your code is vectorized.

To solve this problem, you have to create data points that approximately lie on a 2D plane, display these points, and display the predictions of your model after tuning the parameters with gradient descent.

**Problem 3** 

What does an average MNIST digit look like? For each i=0,1,...,9, compute the average of digit i and display it.  More precisely, you have 
add all the images of the digit i together and divide it by the number of times the digit i occurs in the data set. 

Use ```tf.keras.datasets``` to load the MNIST digits dataset.

---

You have to post the links to your editable notebooks in Webcourses, save the notebooks in your GitHub course repo in the folder ```HW_2```.  Make sure that you follow the style guide for Python code.  Also, you should use text cells with markup headings to break up your notebooks into logical units. Always cite all sources used.
