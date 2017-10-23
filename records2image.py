import tensorflow as tf
import numpy as np
from PIL import Image

import os



# cwd = os.getcwd()

root = "images"
labels_path = 'labels.txt'

fileNameQue = tf.train.string_input_producer(["outputs/hmdb51.tfrecords"])
reader = tf.TFRecordReader()
key,value = reader.read(fileNameQue)
features = tf.parse_single_example(value,features={ 'label': tf.FixedLenFeature([], tf.int64),
                                           'img' : tf.FixedLenFeature([], tf.string),})

img = tf.decode_raw(features["img"], tf.uint8)
label = tf.cast(features["label"], tf.int32)
img  =tf.reshape(img, (299, 299, 3))
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000,
                                                allow_smaller_final_batch=True)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):
        # imgArr, imgLabel = sess.run([img, label])
        # print (imgArr.shape, imgLabel)
        imgArr, imgLabel = sess.run([img_batch, label_batch])
        print(imgArr.shape, imgLabel)

    coord.request_stop()
    coord.join(threads)