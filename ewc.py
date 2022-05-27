#%%
# source: https://seanmoriarity.com/2020/10/18/continual-learning-with-ewc/
# ewc: elastic weight consolidation
from numpy import gradient
import tensorflow as tf
import tensorflow_datasets as tfds 
# from tensorflow.python.keras.datasets import mnist
# from tensorflow.python.keras.api import keras
from tensorflow import keras 
import os
import matplotlib.pyplot as plt
#%%
(mnist_train, mnist_test), ds_info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True)

#%% data preparation
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label 

def transform_labels(image, label):
    # for mnist dataset labels: [0-9] (total 10 values) will be transformed to [0-4] (total 10 values)
    return image, tf.math.floor(label / 2)

def prepare(ds, shuffle=True, batch_size=32, prefetch=True):
    ds = ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(transform_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(ds_info.splits['train'].num_examples) if shuffle else ds 
    ds = ds.cache() 
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE) if prefetch else ds
    return ds

def split_tasks(ds, predicate):
    return ds.filter(predicate), ds.filter(lambda img, label: not predicate(img, label))

def evaluate(model, test_set):
  acc = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
  for i, (imgs, labels) in enumerate(test_set):
    preds = model.predict_on_batch(imgs)
    acc.update_state(labels, preds)
  return acc.result().numpy()

multi_task_train, multi_task_test = prepare(mnist_train), prepare(mnist_test)
task_A_train, task_B_train = split_tasks(mnist_train, lambda img, label: label % 2 ==0)
task_A_train, task_B_train = prepare(task_A_train), prepare(task_B_train)
task_A_test, task_B_test = split_tasks(mnist_test, lambda img, label: label % 2 == 0)
task_A_test, task_B_test = prepare(task_A_test), prepare(task_B_test)

# %% multi-task training
multi_task_model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(5)
])
multi_task_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
multi_task_model.fit(multi_task_train, epochs=6)
#%% accuracy on multi-task 
print("Task A accuracy after training on Multi-Task Problem: {}".format(evaluate(multi_task_model, task_A_test)))
print("Task B accuracy after training on Multi-Task Problem: {}".format(evaluate(multi_task_model, task_B_test)))
# %% now to demonstrate difficulty in continual learning, create a new model and train it on task A
basic_cl_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense((5))
])
basic_cl_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
basic_cl_model.fit(task_A_train, epochs=6)
#%% Next evaluate the model on task A
print("Task A accuracy after training only on task A: {}".format(evaluate(basic_cl_model, task_A_test)))
# %% train model on task B 
basic_cl_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
basic_cl_model.fit(task_B_train, epochs=6)
#%% Next evaluate model on task B and task A
print("Task B accuracy after training trained model on Task B: {}".format(evaluate(basic_cl_model, task_B_test)))
print("Task A accuracy after training trained model on Task B: {}".format(evaluate(basic_cl_model, task_A_test)))
print('Notice how the model easily solves task B; however, it looses nearly all of it’s knowledge about task A')
# %% train with l-2 penalty keeping theta_A closer to new weights
def l2_penalty(theta, theta_A):
  penalty = 0
  for i, theta_i in enumerate(theta):
    _penalty = tf.math.reduce_sum((theta_i - theta_A[i])**2)
    penalty += _penalty
  return 0.5 * penalty

def train_with_l2(model, task_A_train, task_B_train, task_A_test, task_B_test, epochs=6):
  """ model: compiled model
  """
  # train on task A and retain a copy of parameters traine on task A
  print("Train on Task A...")
  model.fit(task_A_train, epochs=epochs) 
  theta_A = {n : p.value() for n, p in enumerate(model.trainable_variables.copy())}
  print(f"Task A accuracy trained on Task A: {evaluate(model, task_A_test)}\n")
  accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
  loss = tf.keras.metrics.SparseCategoricalCrossentropy('loss')
  print('\n Train on Task B with l_2 penalty...')
  for epoch in range(epochs):
    accuracy.reset_states()
    loss.reset_states()
    for batch, (imgs, labels) in enumerate(task_B_train):
      with tf.GradientTape() as tape:
        preds = model(imgs)
        total_loss = model.loss(labels, preds) + l2_penalty(model.trainable_variables, theta_A)
      grads = tape.gradient(total_loss, model.trainable_variables)
      model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
      accuracy.update_state(labels, preds)
      loss.update_state(labels, preds)
      print(f"\rEpoch: {epoch+1}/{epochs}, Batch: {batch+1}, Loss: {loss.result().numpy():.3f}, Accuracy: {accuracy.result().numpy():.3f}", 
      flush=True, end='')
    print("")
  print("Task B accuracy after training trained model on Task B: {}".format(evaluate(model, task_B_test)))
  print("Task A accuracy after training trained model on Task B: {}".format(evaluate(model, task_A_test)))
  print("Notice that the model does not perform well on Task B, our goal is to improve Task B accuracy without scrificing Task A accuarcy much.")
#%%
l2_model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(5)
])
l2_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
train_with_l2(l2_model, task_A_train, task_B_train, task_A_test, task_B_test)
# %% Lets now try EWC penalty 
def compute_precision_matrix(model, task_A_train, num_batches=1):
  """
  model: model fitted to Task A
  """
  task_A_train = task_A_train.repeat()
  precision_metrices = {n:tf.zeros_like(p.value()) for n, p in enumerate(model.trainable_variables)}
  for _, (image, _) in enumerate(task_A_train.take(num_batches)):
    # we need gradients of the model parameters 
    with tf.GradientTape() as tape:
      # get model predictions for each image 
      preds = model(image)
      # Get the log-likelihood of each preds 
      ll = tf.nn.log_softmax(preds)
    # attach gradients of ll to ll_grads 
    ll_grads = tape.gradient(ll, model.trainable_variables)
    # compute F_i as mean of gradient squared 
    for j, g in enumerate(ll_grads):
      precision_metrices[j] += tf.math.reduce_mean(g ** 2, axis=0 ) / num_batches
  return precision_metrices

def compute_elastic_penalty(F, theta, theta_A, alpha=25):
  penalty = 0.0 
  for i, theta_i in enumerate(theta):
    _penalty = tf.math.reduce_sum(F[i] * (theta_i - theta_A[i]) ** 2 )
    penalty += _penalty 
  return 0.5 * alpha * penalty 

def ewc_loss(labels, preds, model, F, theta_A):
  loss_b = model.loss(labels, preds)
  # compute elastic penalty 
  penalty = compute_elastic_penalty(F, model.trainable_variables, theta_A)
  return loss_b + penalty 
#%%
def train_with_ewc(model, task_A_train, task_B_train, task_A_test, task_B_test, epochs=3):
  # first fit on Task A and retain a copy of model-parameters trained on Task A
  model.fit(task_A_train, epochs=epochs)
  theta_A = {n:p.value() for n, p in enumerate(model.trainable_variables.copy())}
  F = compute_precision_matrix(model, task_A_train, num_batches=1000)
  print("Task A accuracy after training on Task A: {}".format(evaluate(model, task_A_test)))
  # setup training loop for Task B with EWC
  accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
  loss = tf.keras.metrics.SparseCategoricalCrossentropy('loss')
  for epoch in range(3*epochs):
    accuracy.reset_states()
    loss.reset_states()
    for batch, (images, labels) in enumerate(task_B_train):
      with tf.GradientTape() as tape:
        # make predictions 
        preds = model(images)
        # compute EWC loss 
        total_loss  = ewc_loss(labels, preds, model, F, theta_A)
      # compute gradients of model's trainable parameters w.r.t. total loss 
      grads = tape.gradient(total_loss, model.trainable_variables)
      # update the model with gradients 
      model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
      # report updated accuracy and loss 
      accuracy.update_state(labels, preds)
      loss.update_state(labels, preds)
      print(f"\rEpoch: {epoch+1} / {3*epochs}, Batch: {batch+1}, Loss: {loss.result().numpy():.3f}, Accuracy: {accuracy.result().numpy():.3f}", 
      flush=True, end=''
      )
    print("")

  print("Task B accuracy after training trained model on Task B: {}".format(evaluate(model, task_B_test)))
  print("Task A accuracy after training trained model on Task B: {}".format(evaluate(model, task_A_test)))
#%%
model_ewc = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(5)
])
model_ewc.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
train_with_ewc(model_ewc, task_A_train, task_B_train, task_A_test, task_B_test, epochs=6)

# %%
