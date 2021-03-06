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

def parse_my_parser(argparse):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default=r'D:\Users\kingdom\Datasets\HMDB51\hmdb51_org_records',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        default=r'D:\Users\kingdom\Datasets\HMDB51\hmdb51_org',
        help='Path to folders of labeled videos.'
    )
    parser.add_argument(
        '--record_dir',
        type=str,
        default=r'D:\Users\kingdom\Datasets\HMDB51\hmdb51_org_records',
        help='Path to folders of labeled video records.'
    )
    parser.add_argument(
        '--split_dir',
        type=str,
        default=r'D:\Users\kingdom\Datasets\HMDB51\testTrainMulti_7030_splits',
        help='Path to folders of split files.'
    )
    parser.add_argument(
        '--split_round',
        type=int,
        default=1,
        help='which round will be processed.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="""\
             How many steps to store intermediate graph. If "0" then will not
             store.\
          """
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=10,  # 1 for attention, 100 for lstm
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=10,  # 1 for attention, -1 for others
        help="""\
          How many images to test on. This test set is only used once, to evaluate
          the final accuracy of the model after training completes.
          A value of -1 causes the entire test set to be used, which leads to more
          stable results across runs.\
          """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=10,  # 1 for attention, 100 for lstm
        help="""\
          How many images to use in an evaluation batch. This validation set is
          used much more often than the test set, and is an early indicator of how
          accurate the model is during training.
          A value of -1 causes the entire validation set to be used, which leads to
          more stable results across training iterations, but may be slower on large
          training sets.\
          """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\
          Whether to print out a list of all misclassified test images.\
          """,
        action='store_true'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='tmp/imagenet',
        help="""\
          Path to classify_image_graph_def.pb,
          imagenet_synset_to_human_label_map.txt, and
          imagenet_2012_challenge_label_map_proto.pbtxt.\
          """
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default=r'D:\Users\kingdom\Datasets\HMDB51\hmdb51_org_bottlenecks',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--aggregated_tensor_name',
        type=str,
        default='aggregated_result',
        help="""\
            The name of the output attention layer in the retrained graph.\
            """
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
          The name of the output classification layer in the retrained graph.\
          """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
          Whether to randomly flip half of the training images horizontally.\
          """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
          A percentage determining how much of a margin to randomly crop off the
          training images.\
          """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
          A percentage determining how much to randomly scale up the size of the
          training images by.\
          """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
          A percentage determining how much to randomly multiply the training image
          input pixels up or down by.\
          """
    )
    parser.add_argument(
        '--use_fp16',
        type=bool,
        default=False,
        help='Train using 16-bit floats instead of 32bit floats'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='inception_v3',
        help="""\
          Which model architecture to use. 'inception_v3' is the most accurate, but
          also the slowest. For faster or smaller models, chose a MobileNet with the
          form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
          'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
          pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
          less accurate, but smaller and faster network that's 920 KB on disk and
          takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
          for more information on Mobilenet.\
          """)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

if __name__ == '__main__':
    import argparse
    FLAGS, unparsed = parse_my_parser(argparse)
    print(FLAGS, unparsed)
