import tensorflow as tf

with tf.device('/gpu:0'):
    a = tf.Variable([[1., 2.]])
    b = tf.constant([[3.], [4.]])
    print(tf.matmul(a, b))
