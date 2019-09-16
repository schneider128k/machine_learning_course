## Problem 1 
[hw_1_problem_1.ipynb](https://colab.research.google.com/drive/1iox1Gm87uqdlRtWriW34zZ5c6sKfRrhx)

## Problem 1 solved with Keras (train only one binary classifier)
[hw_1_problem_1_keras.ipynb](https://colab.research.google.com/drive/1EeUTh-_qNbuQZDLqttYwcKf3zhsAuO-J)

## Problem 2

Note that there is only minimal difference between the code for Problem 1 and the code for Problem 2. You only have to delete ```sigma_prime(z)``` in the lines where you compute ```grad_weights``` and ```grad_bias``` in the method ```fit``` of the class ```Model```. The advantage of the binary crossentropy function is that neurons learn faster.  I have commented out the corresponding lines and add the lines without ```sigma_prime(z)```.

[hw_1_problem_2_binary_crossentropy.ipynb](https://colab.research.google.com/drive/1s0WCKT7baDk1-WSStYgqRw6LWlcTvzSu)

## Problem 3

This is an implementation of softmax with categorical crossentropy from scratch. The code is fully-vectorized code, that is, there is no for loop to iterate over the elements of a mini-batch. To achieve this, I used ```np.tensordot```.

[numpy docs: np.tensordot](https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.tensordot.html)

[TensorFlow docs: tf.tensordot](https://www.tensorflow.org/api_docs/python/tf/tensordot)

[hw_1_problem_3_softmax.ipynb](https://colab.research.google.com/drive/1O7W_t_WMT796Q9FgStxt6PXd7pJAJ621)

I want to clean up the code some more. 

## Problem 4

It should be better to increase ```batch_size``` and ```epochs```. Increasing ```batch_size``` makes training faster because the GPU can process a batch faster, then processing the training examples seperately. Increasing ```batch_size``` decreases the number of times the gradient is updated. Therefore, ```epochs``` should be increased as well.

I have left the hyperparameters the same so we can better compare with the other problems.

[hw_1_problem_4_softmax_keras.ipynb](https://colab.research.google.com/drive/1RwhFCd6Oaw9fq57MVnMJR-fke7bdvUK8)

## Problem 5 (only compute number of connected components and download num_cc arrays to local file system)

It takes a few minutes to compute the number of connected regions (number of white regions) for the train and test images. I decided to compute it only once and download the num_cc arrays to the local file system.

[hw_1_problem_5_only_connected_components.ipynb](https://colab.research.google.com/drive/1RSeZXKBIRMCSK4XPIjJY1ziWb1IjIfx_)

## Problem 5 (uploads num_arrays from local file system)

Observe that the accuracy is better compared to the previous solution in Problem 4 that does not use the additional information about the number of white regions. You can see this by comparing the diagonals of the confusion matrices in Problems 4 and 5. You will see that the diagonal entries in Problem 5 are larger.

[hw_1_problem_5.ipynb](https://colab.research.google.com/drive/1VEWtPJP_iuY4UvTPt6NxRAsjC8SqJd12)
