# Copyright 2017 LIU XIAOYU. All Rights Reserved.
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
import os
import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'data_%s.tfrecord'

SPLITS_TO_SIZES = {'train': 29697, 'ped': 10150, 'ft1': 11476}

LABELS = {
    'none' : (0, 'Background'),
    'car' : (1, 'Vehicle'),
}

_NUM_CLASSES = 1

_ITEMS_TO_DESCRIPTIONS = {
    'image/encoded': 'A [700 x 800 x 11] image.',
    'image/height': 'Image height',
    'image/width': 'Image width',
    'image/channels': 'Image channels',
    'object/bbox': 'Object Bounding box',
    'object/bbox/label': 'label',
}


def get_split(split_name, dataset_dir, file_pattern = None, reader = None):
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError("split name %s was not recognized." % split_name)
    
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader
    
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value = ''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value = 'raw'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/orientation': tf.VarLenFeature(dtype=tf.float32),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'channels': slim.tfexample_decoder.Tensor('image/channels'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'
        ),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/orientation': slim.tfexample_decoder.Tensor('image/orientation'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = None

    return slim.dataset.Dataset(
        data_sources = file_pattern,
        reader = reader,
        decoder = decoder,
        num_samples = SPLITS_TO_SIZES[split_name],
        items_to_descriptions = _ITEMS_TO_DESCRIPTIONS,
    )
# _ITEMS_TO_DESCRIPTIONS = {
#     'image': 'A [600 x 600 x 11] image.',
#     'image/shape': 'Image height',
#     'object/bbox': 'Object Bounding box',
#     'object/bbox/label': 'label',
# }
def get_split2(split_name, dataset_dir, file_pattern = None, reader = None):
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError("split name %s was not recognized." % split_name)
    
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader
    
    keys_to_features = {
        'image': tf.VarLenFeature(dtype=tf.float32),
        'image/shape': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Tensor('image', shape_keys='image/shape'),

        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'
        ),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = None

    return slim.dataset.Dataset(
        data_sources = file_pattern,
        reader = reader,
        decoder = decoder,
        num_samples = SPLITS_TO_SIZES[split_name],
        items_to_descriptions = _ITEMS_TO_DESCRIPTIONS,
    )