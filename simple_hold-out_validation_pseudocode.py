# Split non-test data into training and validation data

train_data = data[num_validation_samples:]
validation_data = data[:num_validation_samples]

model = get_model()
model.train(training_data)

validation_score = model.evaluate(validation_data)

# At this point you can tune your model,
# retrain it, evaluate it, tune it again ...

# Once you've tuned your hyperparameters,
# it's common to train your final model
# from scratch on all non-test data available.

model = get_model()
model.train(data)

# Test your model on test data
test_score = model.evaluate(test_data)
