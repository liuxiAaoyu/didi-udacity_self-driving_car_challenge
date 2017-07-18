# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts KITTI data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'image_2'. Similarly, bounding box annotations are supposed to be
stored in the 'label_2'

This TensorFlow script converts the training and validation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing PNG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'PNG'

    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import os.path
import sys
import random
import pickle

import numpy as np
import tensorflow as tf

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

UDACITY_LABELS = {
    'none': (0, 'Background'),
    'Car': (1, 'Vehicle'),
    'Truck': (2, 'Vehicle'),
    'Pedestrian': (3, 'Persion'),
}
UDACITY_LIST = ['Background','Vehicle','Vehicle','Persion']

def _jpeg_image_shape(image_data, sess, decoded_jpeg, inputs):
    rimg = sess.run(decoded_jpeg, feed_dict={inputs: image_data})
    return rimg.shape


def _process_image(data, f_jpeg_image_shape):
    """Process a image and annotation file.

    Args:
      data: roidb info.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the JPEG image file.
    filename = data['filepath']
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    shape = list(f_jpeg_image_shape(image_data))

    # Get object annotations.
    labels = []
    labels_text = []
    bboxes = []

    for i in range(len(data['gt_classes'])):
        labels.append(data['gt_classes'][i])
        labels_text.append(UDACITY_LIST[data['gt_classes'][i]].encode('ascii'))

        # bbox.
        bboxes.append((float(data['boxes'][i][0]) / shape[1],
                        float(data['boxes'][i][1]) / shape[0],
                        float(data['boxes'][i][2]) / shape[1],
                        float(data['boxes'][i][3]) / shape[0]
                        ))
    

    return (image_data, shape, labels, labels_text, bboxes)


def _convert_to_example(image_data, shape, labels, labels_text,
                        bboxes):
    """Build an Example proto for an image example.

    Args:
      image_data: string, PNG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    # Transpose bboxes, dimensions and locations.
    bboxes = list(map(list, zip(*bboxes)))
    # Iterators.
    it_bboxes = iter(bboxes)

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data),
            'object/label': int64_feature(labels),
            'object/label_text': bytes_feature(labels_text),
            'object/bbox/xmin': float_feature(next(it_bboxes, [])),
            'object/bbox/ymin': float_feature(next(it_bboxes, [])),
            'object/bbox/xmax': float_feature(next(it_bboxes, [])),
            'object/bbox/ymax': float_feature(next(it_bboxes, [])),
            }))
    return example


def _add_to_tfrecord(name, tfrecord_writer, f_png_image_shape):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    l_data = _process_image( name, f_png_image_shape)
    example = _convert_to_example(*l_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name):
    return '%s/%s.tfrecord' % (output_dir, name)


def run(roidb_dir, output_dir, name='udacity_train', shuffling=True):
    """Runs the conversion operation.

    Args:
      dataset_dir: The roidb directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(roidb_dir):
        print('ROI data files not exist.')
        return

    tf_filename = _get_output_filename(output_dir, name)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        # return
    # Dataset filenames, and shuffling.
    with open(roidb_dir, 'rb') as fid:
        roidb = pickle.load(fid)
    
    fileIndexes = list(range(len(roidb)))
    if shuffling:
        random.shuffle(fileIndexes)
    #fileIndexes = list(range(5))
    
    # JPEG decoding.
    inputs = tf.placeholder(dtype=tf.string)
    decoded_jpeg = tf.image.decode_jpeg(inputs)
    with tf.Session() as sess:

        # Process dataset files.
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for i, fileIndex in enumerate(fileIndexes):
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(fileIndexes)))
                sys.stdout.flush()

                data = roidb[fileIndex]
                _add_to_tfrecord( data, tfrecord_writer,
                                 lambda x: _jpeg_image_shape(x, sess, decoded_jpeg, inputs))
        print('\nFinished converting the UDACITY dataset!')

