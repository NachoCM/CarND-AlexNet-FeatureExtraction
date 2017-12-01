import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from alexnet import AlexNet
import sys

# TODO: Load traffic signs data.
sign_names = pd.read_csv('signnames.csv')
nb_classes = 43
with open('train.p', mode='rb') as f:
    data = pickle.load(f)
nsamples=len(data['labels'])
print("Data file contains "+str(nsamples)+" samples.")



# TODO: Split data into training and validation sets.
ntrain=int(nsamples*0.15)
nvalid=int(nsamples*0.05)
ntest=nsamples-ntrain-nvalid
X_train, y_train = data['features'][0:ntrain], data['labels'][0:ntrain]
X_valid, y_valid = data['features'][ntrain:ntrain + nvalid], data['labels'][ntrain:ntrain + nvalid]
X_test, y_test = data['features'][ntrain+nvalid:nsamples], data['labels'][ntrain+nvalid:nsamples]

def preprocess(x):
    x=(x-np.mean(x,axis=(1,2,3),keepdims=True))/np.std(x,axis=(1,2,3),keepdims=True)

preprocess(X_train)
preprocess(X_valid)
preprocess(X_test)

print("Training set with "+str(len(X_train))+" samples")
print("Validation set with "+str(len(X_valid))+" samples")
print("Test set with "+str(len(X_test))+" samples")
# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None), name='y')
one_hot_y = tf.one_hot(y, nb_classes)
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc_signs_W = tf.Variable(tf.truncated_normal(shape,stddev=0.01))
fc_signs_b = tf.Variable(tf.zeros(nb_classes))

logits = tf.nn.xw_plus_b(fc7, fc_signs_W,fc_signs_b)
probs = tf.nn.softmax(logits)
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
EPOCHS=20
BATCH_SIZE=128
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# TODO: Train and evaluate the feature extraction model.

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    #sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



num_examples = len(X_train)

print("Training...")
print()
for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        sys.stdout.write("\r EPOCH {} ...".format(i + 1)+" "+str(offset)+"/"+str(num_examples))
        sys.stdout.flush()

    training_accuracy = evaluate(X_train, y_train)
    validation_accuracy = evaluate(X_valid, y_valid)
    print("EPOCH {} ...".format(i + 1))
    print("Training Accuracy = {:.3f}".format(training_accuracy))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print()