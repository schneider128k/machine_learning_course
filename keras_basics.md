
# Keras basics

## Layers: the building blocks of deep learning

- The fundamental data structure in neural networks is the **layer**.
- A layer is a data-processing module that takes as input one or more tensors and that outputs one or more tensors.
- Some layers are stateless, but more frequently layers have a state: the layers **weights**, one or several tensors learned with stochastic gradient descent, which together contain the network's **knowledge**.

---

- Different layers are appropriate for different tensor formats and different types of data processing.

- Simple vector data, stored in 2D tensors of shape ```(samples, features)``` is often processed by **densely connected layers**, 
also called **fully connected layers** (the ```Dense``` class in Keras).

- Sequence data, stored in 3D tensors of shape ```(samples, timesteps, features)```, is typically processed by **recurrent** layers 
such as **long-short term memory (LSTM)** layer.

- Image data, stored in 4D tensors, is usually processed by 2D **convolutional** layers ```Con2D```. 

Links to Keras documentation:

- [Core layers](https://keras.io/layers/core/)
- [Convolutional layers](https://keras.io/layers/convolutional/)
- [Recurrent layers](https://keras.io/layers/recurrent/)

---

- You can think of layers as LEGO bricks of deep learning.
- Building deep-learning models in Keras is done by combining compatible layers to form useful data-processing pipelines.
- Layer compatibility means that every layer will only accept input tensors of a certain shape and will return output tensors of a certain shape.
- When using Keras, you don't have to worry about compatibility, because the layers you add to your model are dynamically built to match the shape of the incoming layer.

---

```
%tensorflow_version 2.x
import tensorflow as tf

# define a sequential model
network = tf.keras.models.Sequential()

# add a dense layer with relu activation function
# it takes as input a batch of vectors of length 28*28 
network.add(tf.keras.layers.Dense(512,
                                  activation='relu',
                                  input_shape=(28 * 28,)))

# no need to specify input_shape for second dense layer 
# it is dense layer with softmax activation function
network.add(tf.keras.layers.Dense(10,
                                  activation='softmax')) 
```

The second layer didn't receive an input shape argument - instead, it automatically inferred its input shape as being the output shape of the first layer.
