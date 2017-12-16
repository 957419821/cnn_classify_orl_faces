import tensorflow as tf
import dataset as ds

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1) 
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
def max_poll_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

sess = tf.InteractiveSession()
dataSet = ds.DataSet()
x = tf.placeholder(tf.float32, [None, dataSet.height*dataSet.width])
print('reshaping x')
x_image = tf.reshape(x, [-1, 112, 92, 1])
y_ = tf.placeholder(tf.float32, [None, 40])
keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_poll_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_poll_2x2(h_conv2)

W_fc1 = weight_variable([28*23*32, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 28*23*32])
h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 40])
b_fc2 = bias_variable([40])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

print("--------------------")
print("training")
print("--------------------")
for i in range(10000):
    batch = dataSet.nextBatch()
    train_accuracy, _= sess.run([accuracy, train_step], feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
    print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
