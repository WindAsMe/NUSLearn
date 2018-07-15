import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# perceptron.py shows the implementation of a single perceptron node

class Perceptron:

    # State variables

    # Current epoch and error
    curr_epoch = 0
    curr_error = 99e99

    # Activation functions

    # Step function centered at 0
    def step(self, x):
        is_greater = tf.greater(x, 0)
        as_float = tf.to_float(is_greater)
        doubled = tf.multiply(as_float, 2)
        return tf.subtract(doubled, 1)

    # Identify function
    def id(self, x):
        return x

# Class initializer
    # Takes number of input nodes including bias, and
    # 0 to use step function, 1 to use identity function

    def __init__(self, num_in, xfer=0):
        self.in_count = num_in

        # Create weights
        self.w = tf.Variable(tf.random_normal([num_in, 1]))

        # Create placeholders for the network inputs and targets
        self.IN = tf.placeholder(tf.float32, [None, num_in])
        self.OUT = tf.placeholder(tf.float32, [None, 1])

        # Assigns the activation function
        if not xfer:
            self.act = self.step
        else:
            self.act = self.id

        # Initialize all session variables
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

# This is just a playful function but it does tell you the number of input nodes and
# the activation function you chose.

    def hello(self):
        print "Hello world!"
        print "Number of input nodes: {}".format(self.in_count)
        if self.act == self.step:
            print "Using step function"
        else:
            print "Using id function"

    # Does a feedforward but without evaluation.  Private function

    def __feedforward(self, x):

        # Does a dot product between the input and weights and calls the
        # activation function.
        return self.act(tf.matmul(x, self.w))

    # Calls feedforward and returns the result
    def predict(self, x):
        return self.sess.run(self.__feedforward(x))

    # Training routine. train_in = input, train_out = targets, alpha = training rate,
    # max_epochs = maximum number of iterations to run, max_error = maximum allowed error,
    # restart=1 causes variables and weights to be re-initialized

    def train(self, train_in, train_out, alpha=0.01, max_epochs=10, max_error=0.01, restart=0):

        # Calculate output
        output = self.__feedforward(self.IN)

        # Calculate error
        error = tf.subtract(self.OUT, output)

        # find mean square error
        mse = tf.reduce_mean(tf.square(error))

        # Calculates change to weights = (error * input)
        delta = tf.matmul(self.IN, error, transpose_a=True)

        # Multiply delta by training rate
        ldelta = tf.multiply(alpha, delta)
    
        # Adds change to weights
        train = tf.assign(self.w, tf.add(self.w, ldelta))

        print "\nStarting training"
        print "=================\n"

        self.curr_epoch = 0
        self.curr_error = 99.9

        if restart:
            self.sess(self.init)

        while self.curr_epoch < max_epochs and self.curr_error > max_error:
            self.curr_epoch = self.curr_epoch + 1
            self.curr_error, _ = self.sess.run([mse, train], feed_dict={self.IN: train_in, self.OUT: train_out})
            print "Epoch {} of {}: Error is {}".format(self.curr_epoch, max_epochs, self.curr_error)

        return self.curr_error


if __name__ == '__main__':
    nn = Perceptron(3)
    T = 1.
    F = -1.
    bias = 1.
    net_in = [[F, F, bias], [F, T, bias], [T, F, bias], [T, T, bias]]
    net_out = [[F], [T], [T], [T]]
    nn.train(net_in, net_out, alpha=0.01, max_epochs=1000, max_error=0.01)

    print nn.predict([[1, 0, bias], [0, 1, bias], [1, 1, bias]])

    # ==========================================================================
    # bias = 1.
    # mul2in = [[i / 10.0, bias] for i in range(10)]
    # mul2out = [[(2 * i + 5) / 10.0] for i in range(10)]
    # # print(mul2in)
    # # print(mul2out)
    # nn2 = Perceptron(2, 1)
    # nn2.train(mul2in, mul2out, alpha=0.1, max_epochs=1000, max_error=0.001)
    # print nn2.predict([[0.75, bias]])
    # nn2.train(mul2in, mul2out, alpha=0.1, max_epochs=1000, max_error=0.00001)
    # print nn2.predict([[0.75, bias]])
    #
    # # ==================================================================
    # # XOR is insoluble in a neuron
    # xorin = [[F, F, bias], [F, T, bias], [T, F, bias], [T, T, bias]]
    # xorout = [[F], [T], [T], [F]]
    # nn3 = Perceptron(3)
    # nn3.train(xorin, xorout, alpha=0.01, max_epochs=100, max_error=0.01)
