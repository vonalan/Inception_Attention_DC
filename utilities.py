#!/usr/bin/env python
# coding=utf-8

import os
import sys

import tensorflow as tf
import cv2

data_path1 = r"records\talk\talk.avi.tfrecords"
data_path2 = r"records\laugh\laugh.avi.tfrecords"
data_path3 = r"records\smile\smiles.avi.tfrecords"

def get_min_frame_of_videos(root, postfix='*.avi'):
    min_frame, max_frame = 300, 0
    filenames = tf.gfile.Glob(os.path.join(root, postfix))
    for file in filenames:
        print(file)
        cap = cv2.VideoCapture(file)
        frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frameCount < min_frame:
            min_frame = frameCount
        if frameCount > max_frame:
            max_frame = frameCount
    return min_frame, max_frame

def padded_batch_test(number):
    dataset = tf.contrib.data.Dataset.range(number)
    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    dataset = dataset.padded_batch(4, padded_shapes=[None])
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

def parse_tf_example_protocol_buffer_messages():
    def _parse_function(example_proto):
        features = {
            'frames': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        data = tf.decode_raw(parsed_features["data"], tf.uint8)
        label = tf.cast(parsed_features["label"], tf.int32)

        data = tf.reshape(data, [-1, 299, 299, 3])
        label = tf.cast(label, tf.int32)
        label = tf.fill([tf.cast(tf.shape(data)[0], tf.int32)], label)

        return data, label

    filename = tf.placeholder(tf.string, shape=[None], name='decodeVIDEOinput')
    dataset = tf.contrib.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.shuffle(10000)
    # dataset = dataset.batch(3)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, filename, next_element

def normal_distribution(x, mu, sigma):
    import math
    pi = 3.1415926
    m1 = 1.0/(math.sqrt(2.0 * pi) * sigma)
    m2 = math.exp(-0.5 * (pow((x - mu), 2)/pow(sigma, 2)))
    print('m1: %f, m2: %f'%(m1, m2))
    return m1 * m2


if __name__ == "__main__":
    # sess = tf.Session()
    #
    # filenames = [data_path2, data_path3, data_path1]
    # for name in filenames:
    #     iterator, filename, next_element = parse_tf_example_protocol_buffer_messages()
    #     sess.run(iterator.initializer, feed_dict={filename:[name]})
    #     data, label = sess.run(next_element)
    #     print(data.shape, label.shape, label)

    mu = 5.0
    sigma = 0.5/3.0
    for i in range(7):
        print(normal_distribution(mu + (i - 3) * sigma, mu, sigma))
