import tensorflow as tf
import os
import preprocessing
from model import CNN_Model

preprocessing.make_folder() # Create a folder to save the resized image
preprocessing.resize() # Resize image to 64 by 64 and change it to grayscale
preprocessing.make_csv() # Create labeled image csv

# hyper parameter
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BATCH_SIZE = 125
NUM_CLASSES = 2
CHECK_POINT_DIR = TB_SUMMARY_DIR = "./tensor_board/"

# read csv
csv_file = tf.train.string_input_producer(["./label.csv"], shuffle = True)
csv_reader = tf.TextLineReader()
_, line = csv_reader.read(csv_file)

image_file, label_decoded = tf.decode_csv(line, record_defaults=[[""],[""]])
image_decoded = tf.image.decode_jpeg(tf.read_file(image_file), channels=1)
image_cast = tf.cast(image_decoded, tf.float32)
image = tf.reshape(image_cast, [IMAGE_WIDTH, IMAGE_HEIGHT, 1]) # 64 by 64 , grayscale

test_batch = int(12500 / BATCH_SIZE)
test_image_list = ['./resize_test/' + file_name for file_name in os.listdir('./resize_test/')]
test_image_reader = tf.WholeFileReader()
test_image_name = tf.train.string_input_producer(test_image_list)
_, value = test_image_reader.read(test_image_name)
test_image_decode = tf.cast(tf.image.decode_jpeg(value, channels=1), tf.float32)
test_image = tf.reshape(test_image_decode, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])

# make batch file
image_batch, label_batch, test_batch_x = tf.train.shuffle_batch([image, label_decoded, test_image], batch_size=BATCH_SIZE, num_threads=4, capacity=50000, min_after_dequeue=10000)

# start session
with tf.Session() as sess :
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    models = CNN_Model(sess, "cnn_model")


    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
    writer.add_graph(sess.graph)
    global_step = 0

    # saver = tf.train.Saver()
    print("Learning Start")

    # train model
    for epoch in range(10) :
        avg_cost = 0
        total_batch = int(25000/BATCH_SIZE)
        print(total_batch)
        avg_accuracy = 0
        for i in range(total_batch) :
            batch_x , batch_y = sess.run([image_batch, label_batch])
            batch_y = batch_y.reshape(BATCH_SIZE, 1)

            accuracy = models.get_accuracy(batch_x, batch_y)
            cost_value, summary, _ = models.train(batch_x, batch_y)

            writer.add_summary(summary, global_step=global_step)
            global_step += 1
            avg_cost += (cost_value / total_batch)
            avg_accuracy += (accuracy / total_batch)


            # print('i:', '%04d' %(i+1), 'Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'train_accuracy = {:.2%}'.format(accuracy))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'train_accuracy ={:.2%}'.format(avg_accuracy))
        with open("./result.txt", "a") as f :
            f.write('Epoch:' + '%04d' % (epoch + 1) + 'cost =' + '{:.9f}'.format(avg_cost) +
                    'train_accuracy = {:.2%}\n'.format(avg_accuracy))
        print("Saving network...")
        models.saver.save(sess, CHECK_POINT_DIR + "/model.ckpt")
    print("Learning Finish")


