#!/usr/bin/python
# Copyright 2017 LIU XIAOYU. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import rospy
import sensor_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from mychallenge.msg import BoundingBox
from mychallenge.msg import MyOption

import tensorflow as tf

slim = tf.contrib.slim

from lidarnets.nets import simplifySSD, np_methods
from lidarnets.preprocessing import simplifySSD_preprocessing, visualization


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU
# MEMORY!!!
gpu_options = tf.GPUOptions(
    allow_growth=True, per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.Session(config=config)


# Input placeholder.

net_shape = (600, 600)
#net_shape = (300, 749)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, orientations_pre, bbox_img = simplifySSD_preprocessing.preprocess_for_eval(
    img_input, None, None, None, net_shape, data_format, resize=simplifySSD_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
params = simplifySSD.SSDNet.default_params
params = params._replace(num_classes=3)
ssd_net = simplifySSD.SSDNet(params)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, orientations, _ = ssd_net.net(
        image_4d, is_training=False, dropout_keep_prob=1., update_feat_shapes=True, reuse=reuse)
#predictions, localisations, _, orientations, _ = ssd_net.net(
#        image_4d, is_training=False, dropout_keep_prob=1., update_feat_shapes=True, reuse=reuse)
_tv=slim.get_variables()
for i in _tv:
    print(i)
# Restore SSD model.
#ckpt_filename = '/home/xiaoyu/logs/ssd_300_kitti./model.ckpt-226057'
ckpt_filename = '/home/xiaoyu/catkin_ws/src/mychallenge/models/lidarmodel2.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
#ckpt_filename = '/home/xiaoyu/catkin_ws/src/mychallenge/models/lidarmodel.ckpt'
#saver.save(isess, ckpt_filename)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


bridge = CvBridge()
fnum = []
fnum.append(0)
pnum = []
pnum.append(0)
# Main image processing routine.


def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=net_shape):
    fnum[0] += 1
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rorientations, rbbox_img = isess.run(
        [image_4d, predictions, localisations, orientations, bbox_img], feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, roriens, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, rorientations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=4, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, roriens, rbboxes = np_methods.bboxes_sort(
        rclasses, rscores, roriens, rbboxes, top_k=400)
    rclasses, rscores, roriens, rbboxes = np_methods.bboxes_nms(
        rclasses, rscores, roriens, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    if rscores.size != 0:
        pnum[0] += 1
    print('+++++++++++++++++++++++')
    #print(rbboxes)
    #print(rscores)
    print("%d / %d" % (pnum[0], fnum[0]))
    return rclasses, rscores, roriens, rbboxes


pub = rospy.Publisher("/BoundingBox/Lidar", BoundingBox, queue_size=10)


def GetFeatureMap(msg, fmt='png'):
    img_stamp = msg.header.stamp.to_nsec()
    results = {}
    heightstr = msg.height
    height = np.frombuffer(heightstr,dtype=np.uint8)
    height = np.reshape(height,(-1,600))
    densitystr = msg.density
    density = np.frombuffer(densitystr,dtype=np.uint8)
    density = np.reshape(density,(-1,600))
    ringstr = msg.ring
    ring = np.frombuffer(ringstr,dtype=np.uint8)
    ring = np.reshape(ring,(-1,600))
    intensitystr = msg.intensity
    intensity = np.frombuffer(intensitystr,dtype=np.uint8)
    intensity = np.reshape(intensity,(-1,600))
    cv_image = np.dstack((height,intensity,ring))

    rclasses, rscores, roriens, rbboxes = process_image(cv_image, 0.5, 0.3)
    print(img_stamp)

    myBoundingBox = BoundingBox()
    myBoundingBox.header.stamp = msg.header.stamp
    myBoundingBox.num = len(rclasses)
    myBoundingBox.type = 0
    for i in range(len(rclasses)):
        myBoundingBox.classes.append(rclasses[i])
        myBoundingBox.scores.append(rscores[i])
        myBoundingBox.ymin.append(np.float32(rbboxes[i][0]))
        myBoundingBox.xmin.append(np.float32(rbboxes[i][1]))
        myBoundingBox.ymax.append(np.float32(rbboxes[i][2]))
        myBoundingBox.xmax.append(np.float32(rbboxes[i][3]))
        myBoundingBox.orientations.append(np.float32(roriens[i]*np.pi))
        print("%d,%f \n %f,%f,%f,%f" % (rclasses[i], rscores[i], rbboxes[
              i][0], rbboxes[i][1], rbboxes[i][2], rbboxes[i][3]))
        print("%f,%f" % ((rbboxes[i][2] - rbboxes[i][0])
                         * 600, (rbboxes[i][3] - rbboxes[i][1]) * 600))
    pub.publish(myBoundingBox)

    visualization.bboxes_draw_on_img(
        cv_image, rclasses, rscores, roriens, rbboxes, visualization.colors_plasma)
    cv_image = np.dstack((cv_image[:,:,1],cv_image[:,:,2],cv_image[:,:,0]))
    cv2.imshow("image_raw", cv_image)
#//1.20914 -1.22553 1.18114 
#//1394.62 1568.65 
#//    //1.20514 -1.22953 1.17914 1399.62 1528.65
def listener():
    #cv2.namedWindow("image_heightmap")

    cv2.namedWindow("image_raw")
    cv2.startWindowThread()
    rospy.init_node('python_node2', anonymous=True)


    #when first time allocate GPU memory it will take about 0.2 second 
    test = np.zeros((600,600,3),dtype=np.uint8)
    process_image(test, 0.5, 0.4)
    
    #rospy.Subscriber('/pointcloud/featureMap',
    #                 sensor_msgs.msg.Image, lambda msg: GetFeatureMap(msg))
    #rospy.Subscriber('/image_raw',
    #                 sensor_msgs.msg.Image, lambda msg: GetImage(msg))
    rospy.Subscriber('/pointcloud/featureMap', MyOption, GetFeatureMap)

    rospy.spin()

if __name__ == "__main__":
    listener()


