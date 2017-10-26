#!/usr/bin/env python
# coding=utf-8

import os
import sys

import tensorflow as tf
import cv2

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

if __name__ == "__main__":
    next_element = padded_batch_test(10)

    sess = tf.Session()
    for i in range(100):
        try:
            print(sess.run(next_element))
        except:
            break
