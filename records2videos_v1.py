import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data_path = r'records/lauph/laugh.avi.tfrecords'  # tfrecords 文件的地址

#图片存放位置
DATA_DIR = 'data/'

#图片信息
IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_CHANNELS = 3
# NUM_TRAIN = 7000
# NUM_VALIDARION = 1144

def read_and_decode(filename_queue):
    feature = {
        'frames': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'data': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }

    #创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    #从文件中读出一个样例
    _,serialized_example = reader.read(filename_queue)
    #解析读入的一个样例
    features = tf.parse_single_example(serialized_example, feature)
    #将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['data'],tf.uint8)
    label = tf.cast(features['label'],tf.int64)

    # image.set_shape([IMG_PIXELS])
    image = tf.reshape(image,[-1, IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS])
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return image,label

# #用于获取一个batch_size的图像和label
# def inputs(data_set,batch_size,num_epochs):
#     # if not num_epochs:
#     #     num_epochs = None
#     # if data_set == 'train':
#     #     file = TRAIN_FILE
#     # else:
#     #     file = VALIDATION_FILE
#     file = data_path
#     with tf.name_scope('input') as scope:
#         filename_queue = tf.train.string_input_producer([file], num_epochs=num_epochs)
#     image,label = read_and_decode(filename_queue)
#     # #随机获得batch_size大小的图像和label
#     # images,labels = tf.train.shuffle_batch([image, label],
#     #     batch_size=batch_size,
#     #     num_threads=64,
#     #     capacity=1000 + 3 * batch_size,
#     #     min_after_dequeue=1000
#     # )
#
#     return image,label

filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
image, label = read_and_decode(filename_queue)

with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(1):
        example, l = sess.run([image,label])#在会话中取出image和label
        # img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        # img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example.shape, l)
    coord.request_stop()
    coord.join(threads)