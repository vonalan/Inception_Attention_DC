import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data_path1 = r"E:\Users\kingdom\Inception_Attention_DC\records\talk\talk.avi.tfrecords"
data_path2 = r"E:\Users\kingdom\Inception_Attention_DC\records\laugh\laugh.avi.tfrecords"
data_path3 = r"E:\Users\kingdom\Inception_Attention_DC\records\smile\smiles.avi.tfrecords"

#图片存放位置
DATA_DIR = 'data/'

#图片信息
IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_CHANNELS = 3
# NUM_TRAIN = 7000
# NUM_VALIDARION = 1144

feature = {
        'frames': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'data': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }

fileNameQue = tf.train.string_input_producer([data_path1, data_path2, data_path3])
reader = tf.TFRecordReader()
key,value = reader.read(fileNameQue)
features = tf.parse_single_example(value,features=feature)

img = tf.decode_raw(features["data"], tf.uint8)
label = tf.cast(features["label"], tf.int32)
label = tf.fill()
img = tf.reshape(img, (-1, 299, 299, 3))
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                 batch_size=30, capacity=2000,
#                                                 min_after_dequeue=1000,
#                                                 allow_smaller_final_batch=True, enqueue_many=False)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    epoch = 0
    for i in range(16):
        imgArr, imgLabel = sess.run([img, label])
        # print (imgArr.shape, imgLabel)
        # imgArr, imgLabel = sess.run([img_batch, label_batch])
        print(epoch, imgArr.shape, imgLabel)
        epoch += 1

    coord.request_stop()
    coord.join(threads)