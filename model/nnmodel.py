import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class NNLayer(ABC):
    def __init__(self, name):
        self.name = name
    @abstractmethod
    def get_value(self):
        pass

    def get_name(self):
        return self.name

class NNPlaceHolder(NNLayer):
    def __init__(self, shape, tftype,name):
        super().__init__(name)
        self.shape = shape
        self.tftype = tftype
        self.placeholder = None

    def initialize(self):
        self.placeholder = tf.placeholder(self.tftype, self.shape, self.name)

    def get_value(self):
        return self.placeholder

class NNWeights:
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name
        self.W = None

    def initialize(self, std_dev=0.1):
        self.W = tf.Variable(tf.truncated_normal(self.shape, stddev=std_dev), name= self.name)
        return self.W

    def get_shape(self):
        return self.shape

    def get_W(self):
        return self.W

    def get_name(self):
        return self.name

class NNBias:
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name
        self.b = None

    def initialize(self):
        self.b = tf.Variable(tf.zeros(self.shape, name= self.name), name = self.name)
        return self.b

    def get_b(self):
        return self.b

    def get_shape(self):
        return self.shape

    def get_name(self):
        return self.name

class NNActivationLayer(NNLayer):
    def __init__(self, weights, input_data, bias, name):
        super().__init__(name)
        self.input_data = input_data
        self.weights = weights
        self.bias = bias
        self.activation = None

    def get_value(self):
        return self.activation

    def initialize(self):
        X = self.input_data.get_value()
        W = self.weights.get_W()
        b = self.bias.get_b()
        Z = tf.matmul(W,X) + b
        self.activation = self.local_activation(Z)
        return self.activation

    @abstractmethod
    def local_activation(self, Z):
        pass

class NNSigmoid(NNActivationLayer):
        def __init__(self,weights, input_data, bias, name):
            super().__init__(weights, input_data, bias, name)

        def local_activation(self, Z):
            self.activation = tf.sigmoid(Z, name=self.name)

class NNSoftmax(NNActivationLayer):
    def __init__(self, weights, input_data, bias, name):
        super().__init__(weights, input_data, bias, name)

    def local_activation(self, Z):
        self.activation = tf.nn.softmax(Z, name=self.name)

class NNRelu(NNActivationLayer):
    def __init__(self, weights, input_data, bias, name):
        super().__init__(weights, input_data, bias, name)

    def local_activation(self, Z):
        self.activation = tf.nn.relu(Z, name=self.name)

class NNSoftmaxLinear(NNActivationLayer):
    def __init__(self, weights, input_data, bias, name):
        super().__init__(weights, input_data, bias, name)

    def local_activation(self, Z):
        self.activation = Z

class NNLoss(ABC):
    def __init__(self, labels, logits, name):
        self.labels = labels
        self.logits = logits
        self.name = name
        self.loss = 0

    def loss(self):
        Y_ = self.logits.get_value()
        Y =  self.labels.get_value()
        cross_entropy = self.local_cross_entropy(Y_, Y)
        return tf.reduce_mean(cross_entropy, name= self.name + "_mean")

    @abstractmethod
    def local_cross_entropy(self, Y_, Y):
        pass

class NNSoftmaxCrossEntropy(NNLoss):
    def __init__(self, labels, logits, name):
        super().__init__(labels, logits, name)

    def local_cross_entropy(self, Y_, Y):
        return tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_)

class NNSparseSoftmaxCrossEntropy(NNLoss):
    def __init__(self, labels, logits, name):
        super().__init__(labels, logits, name)

    def local_cross_entropy(self, Y_, Y):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=Y_)

class NNSigmoidCrossEntropy(self, Y_, Y):
    def __init__(self, labels, logits, name):
        super().__init__(labels, logits, name)

    def local_cross_entropy(self, Y_, Y):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_)


c
# class NNSoftmaxCrossEntropy(NNActivationLayer):
#     def __init__(self, weights, input_data, bias, name, labels):
#         super().__init__(weights, input_data, bias, name)
#         self.labels = labels
#
#     def get_labels(self):
#         return self.labels
#
#     def local_activation(self, Z):
#         self.activation = tf.nn.softmax_cross_entropy_with_logits(logits=Z, name=self.name)
#
#     def get_crossentropy(self):
#         return self.activation


# class NNSigmoid(NNLayer):
#     def __init__(self, weights, input_data, bias, name):
#         super().__init__(name)
#         self.input_data = input_data
#         self.weights = weights
#         self.bias = bias
#         self.sigmoid = None
#
#     def initialize(self):
#         X = self.input_data.get_value()
#         W = self.weights.get_W()
#         b = self.bias.get_b()
#         logits = tf.matmul(W,X) + b
#         self.sigmoid = tf.sigmoid(logits, name=self.name)
#         return self.sigmoid
#
#     def get_value(self):
#         return self.sigmoid
#
#
# class NNSoftmax(NNLayer):
#     def __init__(self, weights, input_data, bias, name):
#         super().__init__(name)
#         self.input_data = input_data
#         self.weights = weights
#         self.bias = bias
#         self.softmax = None
#
#     def initialize(self):
#         X = self.input_data.get_value()
#         W = self.weights.get_W()
#         b = self.bias.get_b()
#         logits = tf.matmul(W, X) + b
#         self.softmax = tf.nn.softmax(logits, name=self.name)
#         return self.softmax
#
#     def get_value(self):
#         return self.softmax
#
# class NNRelu(NNLayer):
#     def __init__(self, weights, input_data, bias, name):
#         super().__init__(name)
#         self.input_data = input_data
#         self.weights = weights
#         self.bias = bias
#         self.softmax = None
#
#     def initialize(self):
#         X = self.input_data.get_value()
#         W = self.weights.get_W()
#         b = self.bias.get_b()
#         features = tf.matmul(W, X) + b
#         self.relu = tf.nn.relu(features, name=self.name)
#         return self.relu
#
#     def get_value(self):
#         return self.relu
#
# class NNFinalLayer(NNLayer):
#     def __init__(self, weights, input_data, bias, name):
#

#class NNSoftmaxCrossEntrotyp
class nn_parameter_initializer:
    def __init__(self, n_l, n_examples, layer):
        self.n_features = n_l
        self.n_examples = n_examples
        self.layer = layer

    def init_weights(self, std_dev=0.1):
        W = tf.Variable(tf.truncated_normal([self.n_features, self.n_examples], stddev=std_dev), name = "W"+ self.layer)
        return W

    def init_bias(self):
        b = tf.Variable(tf.zeros([self.n_examples]), name="b"+self.layer)
        return b

    def initialize(self):
        W = self.init_weights()
        b = self.init_bias()
        return (W,b)

class Tweet3LayerNeuralNetwork:
    def __init__(self, layer1, layer2, layer3, train_file, dev_file, test_file):
        #self.layer_data = layer_data
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

    def buildNN(self):

        self.X = tf.placeholder(tf.int32, shape())

        pass

    def train(self):
        pass

    def run_dev(self):
        pass
    def run_test(self):
        pass

    def classify(self):
        pass