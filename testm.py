import tensorflow as tf
import os
'''
User ID: t0916129
Name (English): Zhong Rui
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MLP:

    # Call in # of input nodes excluding bias, number of hidden nodes and
    # number of output nodes. act=0 gives you sigmoid activation, act=1 gives
    # you tanh activation

    def __init__(self, num_in, num_hid1, num_hid2, num_out, act=0):

        # The INPUT and TARGET placeholders
        self.INPUT = tf.placeholder(tf.float32, [None, num_in])
        self.TARGET = tf.placeholder(tf.float32, [None, num_out])

        if act == 0:
            self.actfn = tf.sigmoid
        else:
            self.actfn = tf.tanh

        # Declare the hidden layer. It receives input from
        # INPUT
        self.hidden_layer1 = tf.layers.dense(inputs=self.INPUT, units=num_hid1, activation=self.actfn)

        self.hidden_layer2 = tf.layers.dense(inputs=self.hidden_layer1, units=num_hid2, activation=self.actfn)
        # Declare the output layer. It receives input from the
        # hidden_layer
        self.output_layer = tf.layers.dense(inputs=self.hidden_layer2, units=num_out, activation=self.actfn)

        # Initialize all variables
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def train(self, train_in, train_target, alpha=0.1, max_epochs=1000, max_error=0.01, restart=0):

        # We use the least-squared error
        loss = tf.losses.mean_squared_error(labels=self.TARGET, predictions=self.output_layer)

        # And gradient descent optimizer
        optimizer = tf.train.GradientDescentOptimizer(alpha)
        train = optimizer.minimize(loss)

        curr_epoch = 0
        curr_error = 99e99

        if restart == 1:
            self.sess.run(self.init)

        while curr_epoch < max_epochs and curr_error > max_error:
            curr_epoch = curr_epoch + 1

            # We just simply use run to minimize the loss
            _, curr_error = self.sess.run([train, loss], feed_dict={self.INPUT: train_in, self.TARGET: train_target})

            print "iteration {} of {}: Loss {}".format(curr_epoch, max_epochs, curr_error)

    def predict(self, net_in):
        return self.sess.run(self.output_layer, feed_dict={self.INPUT: net_in})

    def predict_and_score(self, net_in, target):
        error = tf.square(self.output_layer - target)
        mse = tf.reduce_mean(error)

        err, out = self.sess.run([mse, self.output_layer], feed_dict={self.INPUT: net_in, self.TARGET: target})
        return err, out


if __name__ == '__main__':
    # The sig mod is no 0 and 1
    bias = 1
    tin = [[i / 10.0, bias] for i in range(10)]
    tout = [[(2 * i + 5) / 10.0] for i in range(10)]
    nn = MLP(2, 8, 8, 1)
    nn.train(tin, tout, alpha=0.5, max_epochs=5000, max_error=0.001)
