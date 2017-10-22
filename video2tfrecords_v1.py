"""Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels.
 Allows to subsequently train a neural network in TensorFlow with the generated tfrecords.
 Due to common hardware/GPU RAM limitations, this implementation allows to limit the number of frames per
 video actually stored in the tfrecords. The code automatically chooses the frame step size such that there is
 an equal separation distribution of the video images. Implementation supports Optical Flow
 (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, math
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import tensorflow as tf


FLAGS = None
FILE_FILTER = '*.avi'
NUM_FRAMES_PER_VIDEO = 15
NUM_VIDEOS_PER_RECORD = 1
DENSE_OPTICAL_FLOW = False
NUM_CHANNELS_VIDEO = 3
WIDTH_VIDEO = 299
HEIGHT_VIDEO = 299

SOURCE = r"E:\Users\kingdom\HMDB51\hmdb51_org"
DESTINATION = r"E:\Users\kingdom\HMDB51\hmdb51_records"
labels_path  = r"E:\Users\kingdom\HMDB51\hmdb51_labels.txt"

labels = []
with open(labels_path) as f:
    lines = f.readlines()
    labels = [line.strip() for line in lines]

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_videos', NUM_VIDEOS_PER_RECORD, 'Number of videos stored in one single tfrecords file')
flags.DEFINE_string('image_color_depth', np.uint8, 'Color depth for the images stored in the tfrecords files. '
                                                          'Has to correspond to the source video color depth. '
                                                   'Specified as np dtype (e.g. ''np.uint8).')
flags.DEFINE_string('source', SOURCE, 'Directory with video files')
flags.DEFINE_string('output_path', DESTINATION, 'Directory for storing tf records')
flags.DEFINE_boolean('optical_flow', DENSE_OPTICAL_FLOW, 'Indictes whether optical flow shall be computed and added as fourth '
                                           'channel. Defaults to False')

def _float_feature(value): 
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_chunks(l, n):
  """Yield successive n-sized chunks from l.
  Used to create n sublists from a list l"""
  for i in range(0, len(l), n):
    yield l[i:i + n]


def getVideoCapture(path):
    cap = None
    if path:
      cap = cv2.VideoCapture(path)
    return cap


def getNextFrame(cap):
  ret, frame = cap.read()
  if ret == False:
    return None

  return np.asarray(frame)


def compute_dense_optical_flow(prev_image, current_image):
  old_shape = current_image.shape
  prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
  current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
  assert current_image.shape == old_shape
  hsv = np.zeros_like(prev_image)
  hsv[..., 1] = 255

  flow = cv2.calcOpticalFlowFarneback(prev_image_gray, current_image_gray, 0.8, 15, 5, 10, 5, 1.5, 0)

  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang*180/np.pi/2
  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def save_video_to_tfrecords(source_path, destination_path, videos_per_file=FLAGS.num_videos, video_filenames=None,
                            dense_optical_flow=False):
  """calls sub-functions convert_video_to_numpy and save_numpy_to_tfrecords in order to directly export tfrecords files
  :param source_path: directory where video videos are stored
  :param destination_path: directory where tfrecords should be stored
  :param videos_per_file: specifies the number of videos within one tfrecords file
  :param dense_optical_flow: boolean flag that controls if optical flow should be used and added to tfrecords
  """
  global NUM_CHANNELS_VIDEO
  assert (NUM_CHANNELS_VIDEO == 3 and (not dense_optical_flow)) or (NUM_CHANNELS_VIDEO == 4 and dense_optical_flow), "correct NUM_CHANNELS_VIDEO"

  if video_filenames is not None:
    filenames = video_filenames
  else:
    filenames = gfile.Glob(os.path.join(source_path, FILE_FILTER))
  if not filenames:
    raise RuntimeError('No data files found.')

  print('Total videos found: ' + str(len(filenames)))

  # filenames_split = list(get_chunks(filenames, videos_per_file))
  for filename in filenames:
      data = convert_video_to_numpy(filename, dense_optical_flow=dense_optical_flow)
      save_numpy_to_tfrecord(data, destination_path, filename)

def save_numpy_to_tfrecord(data, destination_path, videoname):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
    :param data: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos, i=number of images, c=number of image
    channels, h=image height, w=image width
    :param name: filename; data samples type (train|valid|test)
    :param fragmentSize: specifies how many videos are stored in one tfrecords file
    :param current_batch_number: indicates the current batch index (function call within loop)
    :param total_batch_number: indicates the total number of batches
    """

    # num_videos = data.shape[0]
    num_frames = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]
    num_channels = data.shape[3]

    temp = videoname.split("\\")
    destination_path = os.path.join(destination_path, temp[-2])
    if not os.path.exists(destination_path):
      os.makedirs(destination_path)

    image = data.astype(FLAGS.image_color_depth)
    image_raw = image.tostring()

    feature = dict()
    feature['frames'] = _int64_feature(num_frames)
    feature['height'] = _int64_feature(height)
    feature['width'] = _int64_feature(width)
    feature['depth'] = _int64_feature(num_channels)
    feature['label'] = _int64_feature(labels.index(temp[-2]))
    feature['data'] = _bytes_feature(image_raw)

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    recordname = os.path.join(destination_path, temp[-1] + '.tfrecords')
    print('Writing', recordname)
    writer = tf.python_io.TFRecordWriter(recordname)
    writer.write(example.SerializeToString())
    writer.close()


def convert_video_to_numpy(filename, dense_optical_flow=False):
  """Generates an ndarray from multiple video files given by filenames.
  Implementation chooses frame step size automatically for a equal separation distribution of the video images.

  :param filenames
  :param type: processing type for video data
  :return if no optical flow is used: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos, i=number of images,
  (h,w)=height and width of image, c=channel, if optical flow is used: ndarray(uint32) of (v,i,h,w,
  c+1)"""
  global NUM_CHANNELS_VIDEO
  # if not filenames:
  #   raise RuntimeError('No data files found.')
  #
  # number_of_videos = len(filenames)

  if dense_optical_flow:
    # need an additional channel for the optical flow with one exception:
    global NUM_CHANNELS_VIDEO
    NUM_CHANNELS_VIDEO = 4
    num_real_image_channel = 3
  else:
    # if no optical flow, make everything normal:
    num_real_image_channel = NUM_CHANNELS_VIDEO

  def video_file_to_ndarray(i, filename):
    image = np.zeros((HEIGHT_VIDEO, WIDTH_VIDEO, num_real_image_channel), dtype=FLAGS.image_color_depth)
    video = np.zeros((NUM_FRAMES_PER_VIDEO, HEIGHT_VIDEO, WIDTH_VIDEO, NUM_CHANNELS_VIDEO), dtype=np.uint32)
    imagePrev = None
    assert os.path.isfile(filename), "Couldn't find video file"
    cap = getVideoCapture(filename)
    assert cap is not None, "Couldn't load video capture:" + filename + ". Moving to next video."

    # compute meta data of video
    frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # returns nan, if fps needed a measurement must be implemented
    # frameRate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    steps = math.floor(frameCount / NUM_FRAMES_PER_VIDEO)
    j = 0
    prev_frame_none = False

    restart = True
    assert not (frameCount < 1 or steps < 1), str(filename) + " does not have enough frames. Moving to next video."

    while restart:
      for f in range(int(frameCount)):
        # get next frame after 'steps' iterations:
        # floor used after modulo operation because rounding module before leads to
        # unhandy partition of data (big gab in the end)
        if math.floor(f % steps) == 0:
          frame = getNextFrame(cap)
          # special case handling: opencv's frame count != real frame count, reiterate over same video
          if frame is None and j < NUM_FRAMES_PER_VIDEO:
            if frame and prev_frame_none: break
            prev_frame_none = True
            # repeat with smaller step size
            steps -= 1
            if steps == 0: break
            print("reducing step size due to error")
            j = 0
            cap.release()
            cap = getVideoCapture(filename)
            # wait for image retrieval to be ready
            cv2.waitKey(3000)
            video.fill(0)
            continue
          else:
            if j >= NUM_FRAMES_PER_VIDEO:
              restart = False
              break
            # iterate over channels
            if frame.ndim == 2:
              # cv returns 2 dim array if gray
              resizedImage = cv2.resize(frame[:, :], (HEIGHT_VIDEO, WIDTH_VIDEO))
            else:
              for k in range(num_real_image_channel):
                resizedImage = cv2.resize(frame[:, :, k], (HEIGHT_VIDEO, WIDTH_VIDEO))
                image[:, :, k] = resizedImage

              if dense_optical_flow:
                # optical flow requires at least two images
                if imagePrev is not None:
                  frameFlow = np.zeros((HEIGHT_VIDEO, WIDTH_VIDEO))
                  frameFlow = compute_dense_optical_flow(imagePrev, image)
                  frameFlow = cv2.cvtColor(frameFlow, cv2.COLOR_BGR2GRAY)
                else:
                  frameFlow = np.zeros((HEIGHT_VIDEO, WIDTH_VIDEO))

                imagePrev = image.copy()

            if dense_optical_flow:
              image_with_flow = image.copy()
              image_with_flow = np.concatenate((image_with_flow, np.expand_dims(frameFlow, axis=2)), axis=2)
              video[j, :, :, :] = image_with_flow
            else:
              video[j, :, :, :] = image
            j += 1
            # print('total frames: ' + str(j) + " frame in video: " + str(f))
        else:
          getNextFrame(cap)

    # print(str(i + 1) + " of " + str(number_of_videos) + " videos processed", filenames[i])

    v = video.copy()
    cap.release()
    return v

  return video_file_to_ndarray(0, filename)



def main(argv):
  save_video_to_tfrecords(FLAGS.source, FLAGS.output_path, FLAGS.num_videos, dense_optical_flow=FLAGS.optical_flow)



if __name__ == '__main__':
  app.run()
