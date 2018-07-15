# Everything is tensor
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0, dtype=tf.float32)
    # In tensorflow, everything is graph
    # Directly add is not working
    total = a + b
    # print(total)
    # print a
    # print b
    # print 'a is {}.'.format(a)
    # print 'b is {}.'.format(b)
    # print 'total is {}.'.format(total)

    sess = tf.Session()
    # result = sess.run({'a': a, 'b': b, 'total': total})
    # print(result)
    # print 'a is {}, b is {}, total is {}.'.format(result['a'], result['b'], result['total'])

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y
    n = sess.run(z, feed_dict={x: 3, y: 4.5})
    v = sess.run(z, feed_dict={x: [1, 3], y: [2, 4]})
    # print '3 + 4.5 is {}'.format(n)
    # print '[1, 3] + [2, 4] is {}'.format(v)

    # Input Layers creation
    x = tf.placeholder(tf.float32, shape=[None, 3])
    # A single Output creation
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)

    # This procedure create the W(weight) randomly
    init = tf.global_variables_initializer()
    sess.run(init)
    # print(init)

    out = sess.run(y, feed_dict={x: [[1, 2, 3], [4, 8, 12]]})
    # print 'Output from dense layer: {}'.format(out)

    # ================================================================
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # The module hasn't be trained
    print 'Predicted values: {}.'.format(sess.run(y_pred))

    # Define the loss function
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    print 'Loss is {}.'.format(sess.run(loss))

    # Optimization the module
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Train
    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print 'Iteration {}: Loss{}'.format(i, loss_value)

    pred = sess.run(y_pred)
    print 'New prediction: {}'.format(pred)