import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
#sys.path.append('../')

from nets import simplifySSD, ssd_common, np_methods
from preprocessing import simplifySSD_preprocessing
from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (600, 600)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(net_shape[0], net_shape[1], 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, orie_pre, bbox_img = simplifySSD_preprocessing.preprocess_for_eval(
    img_input, None, None, None, net_shape, data_format, resize=simplifySSD_preprocessing.Resize.NONE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = simplifySSD.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, orientations, _ = ssd_net.net(image_4d, update_feat_shapes=True, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '/home/xiaoyu/logs/ssd_300_kitti/model.ckpt-108331'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

print(dict(ssd_net.params._asdict()))

def process_image(img, select_threshold=0.1, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rorientations, rbbox_img = isess.run([image_4d, predictions, localisations, orientations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rorie, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, orientations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# path = '../demo/'
# image_names = sorted(os.listdir(path))

# img = mpimg.imread(path + image_names[-1])
img = cv2.imread('/home/xiaoyu/Documents/1DIDIUDA/tf_testprocess/a1.png')
rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)