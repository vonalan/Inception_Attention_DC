import tensorflow as tf
import numpy as np
from PIL import Image

import os



# cwd = os.getcwd()

root = "images"
labels_path = 'labels.txt'

labels = []
with open(labels_path) as f:
    lines = f.readlines()
    labels = [line.strip() for line in lines]

TFwriter = tf.python_io.TFRecordWriter("outputs/hmdb51.tfrecords")

for className in os.listdir(root):
    # label = int(className[1:])
    label = className
    classPath = root+"/"+className+"/"
    # print(className, classPath)
    for parent, dirnames, filenames in os.walk(classPath):
        for filename in filenames:
            imgPath = classPath+"/"+filename
            print (imgPath)
            img = Image.open(imgPath)
            img = img.resize((299,299))
            print (img.size,img.mode)
            imgRaw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[labels.index(label)])),
                "img":tf.train.Feature(bytes_list = tf.train.BytesList(value=[imgRaw]))
            }) )
            TFwriter.write(example.SerializeToString())

TFwriter.close()