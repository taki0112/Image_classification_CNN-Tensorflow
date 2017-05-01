import tensorflow as tf


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BATCH_SIZE = 125
NUM_CLASSES = 2

class CNN_Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name) as variable_scope:
            self.training = tf.placeholder(tf.bool)
            # make placeholder
            self.X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name="X")
            self.Y = tf.placeholder(tf.int32, [BATCH_SIZE, 1], name="Y")

            self.Y_one_hot = tf.one_hot(self.Y, NUM_CLASSES, name="Y_one_hot")
            self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1, NUM_CLASSES], name="Y_one_hot")

            # make CNN model
            batch_norm1 = tf.layers.batch_normalization(inputs=self.X, training=self.training, name="batch_norm1")
            conv1 = tf.layers.conv2d(inputs=batch_norm1, filters=32, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, name="conv1")
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2, name="pool1")
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training, name="dropout1")

            batch_norm2 = tf.layers.batch_normalization(inputs=dropout1, training=self.training, name="batch_norm2")
            conv2 = tf.layers.conv2d(inputs=batch_norm2, filters=64, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, name="conv2")
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2, name="pool2")
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training, name="dropout2")

            batch_norm3 = tf.layers.batch_normalization(inputs=dropout2, training=self.training, name="batch_norm3")
            conv3 = tf.layers.conv2d(inputs=batch_norm3, filters=128, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, name="conv3")
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2, name="pool3")
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training, name="dropout3")

            batch_norm4 = tf.layers.batch_normalization(inputs=dropout3, training=self.training, name="batch_norm4")
            conv4 = tf.layers.conv2d(inputs=batch_norm4, filters=256, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, name="conv4")
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], padding="SAME", strides=2, name="pool4")
            dropout4 = tf.layers.dropout(inputs=pool4, rate=0.7, training=self.training, name="dropout4")

            batch_norm5 = tf.layers.batch_normalization(inputs=dropout4, training=self.training, name="batch_norm5")
            conv5 = tf.layers.conv2d(inputs=batch_norm5, filters=512, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, name="conv5")
            pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], padding="SAME", strides=2, name="pool5")
            dropout5 = tf.layers.dropout(inputs=pool5, rate=0.7, training=self.training, name="dropout5")

            batch_norm6 = tf.layers.batch_normalization(inputs=dropout5, training=self.training, name="batch_norm6")
            conv6 = tf.layers.conv2d(inputs=batch_norm6, filters=1024, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu, name="conv6")
            pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], padding="SAME", strides=2, name="pool6")
            dropout6 = tf.layers.dropout(inputs=pool6, rate=0.7, training=self.training, name="dropout6")

            flat = tf.contrib.layers.flatten(dropout6)
            final_batch_norm = tf.layers.batch_normalization(inputs=flat, training=self.training, name="final_batch_norm")
            fully_conn = tf.layers.dense(inputs=final_batch_norm, units=625, activation=tf.nn.relu, name="fully_conn")
            final_dropout = tf.layers.dropout(inputs=fully_conn, rate=0.5, training=self.training, name="final_dropout")

            # make logits
            self.logits = tf.layers.dense(inputs=final_dropout, units=2, name="logits")
        # make cost
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y_one_hot), name="cost")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        self.saver = tf.train.Saver()

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self. Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.summary = tf.summary.merge_all()


    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.summary, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def predict(self, x_data, training=False):
        return self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.X: x_data, self.training: training})