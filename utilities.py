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

if __name__ == "__main__":
    root = r"E:\Users\kingdom\HMDB51\hmdb51_org"
    min_frame, max_frame = get_min_frame_of_videos(root)
    print(min_frame, max_frame)

