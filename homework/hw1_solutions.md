## Problem 1 
[hw_1_problem_1.ipynb](https://colab.research.google.com/drive/1iox1Gm87uqdlRtWriW34zZ5c6sKfRrhx)

## Problem 1 solved with Keras (train only one binary classifier)
[hw_1_problem_1_keras.ipynb](https://colab.research.google.com/drive/1EeUTh-_qNbuQZDLqttYwcKf3zhsAuO-J)

## Problem 2

Note that there is only minimal difference between the code for Problem 1 and the code for Problem 2. You only have to delete ```sigma_prime(z)``` in the lines where you compute ```grad_weights``` and ```grad_bias``` in the method ```fit``` of the class ```Model```. The advantage of the binary crossentropy function is that neurons learn faster.  I have commented out the corresponding lines and add the lines without ```sigma_prime(z)```.

[hw_1_problem_2_binary_crossentropy.ipynb](https://colab.research.google.com/drive/1s0WCKT7baDk1-WSStYgqRw6LWlcTvzSu)

## Problem 3

## Problem 4

It should be better to increase ```batch_size``` and ```epochs```. Increasing ```batch_size``` makes training faster because the GPU can process an batch faster, then processing the training examples individually. Increasing ```batch_size``` decreases the number of times the gradient is updated. Therefore, ```epochs``` should be increased as well.

[hw_1_problem_4_softmax_keras.ipynb](https://colab.research.google.com/drive/1RwhFCd6Oaw9fq57MVnMJR-fke7bdvUK8)
