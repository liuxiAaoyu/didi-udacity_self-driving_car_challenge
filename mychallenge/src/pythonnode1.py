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


import tensorflow as tf

slim = tf.contrib.slim

from nets import ssd_vgg_300, np_methods
from preprocessing import ssd_vgg_preprocessing, visualization


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.8)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.Session(config=config)

# Input placeholder.

net_shape = (300, 400)
#net_shape = (300, 749)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
params = ssd_vgg_300.SSDNet.default_params
params = params._replace(num_classes=4)
ssd_net = ssd_vgg_300.SSDNet(params)
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, update_feat_shapes=True, reuse=reuse)

# Restore SSD model.
ckpt_filename = '/home/xiaoyu/catkin_ws/src/mychallenge/models/imagemodel.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

fnum=[]
fnum.append(0)
pnum=[]
pnum.append(0)
# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=net_shape):
    fnum[0]+=1
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img], feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=4, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    if rscores.size!=0:
        pnum[0]+=1
    print('+++++++++++++++++++++++')
    print(rbboxes)
    print(rscores)
    print("%d / %d"%(pnum[0],fnum[0]))
    return rclasses, rscores, rbboxes



heightmaps=[]
bridge = CvBridge()
pub = rospy.Publisher("/BoundingBox/Radar", BoundingBox, queue_size=10)


def GetImage(msg,fmt='png'):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        #cv2.imshow("image_raw", cv_image)
    except CvBridgeError as e:
        print(e)
    cv_image = cv_image[int(cv_image.shape[0]/2)-200:-200,:,:]
 
    rclasses, rscores, rbboxes =  process_image(cv_image,0.8,0.4)

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
        print("%d,%f \n %f,%f,%f,%f"%(rclasses[i],rscores[i],rbboxes[i][0],rbboxes[i][1],rbboxes[i][2],rbboxes[i][3]))
    pub.publish(myBoundingBox)
    visualization.bboxes_draw_on_img(cv_image, rclasses, rscores, rbboxes, visualization.colors_plasma)
    cv2.imshow("image_raw", cv_image)


def listener():
    #cv2.namedWindow("image_heightmap")

    cv2.namedWindow("image_raw")
    cv2.startWindowThread()
    rospy.init_node('mynode', anonymous=True)

    #when first time allocate GPU memory it will take about 0.2 second 
    test = np.zeros((300,400,3),dtype=np.uint8)
    process_image(test, 0.5, 0.4)
    #rospy.Subscriber('/heightmap/pointcloud',
    #                 sensor_msgs.msg.Image, lambda msg: GetHeightMap(msg))
    rospy.Subscriber('/image_raw',
                     sensor_msgs.msg.Image, lambda msg: GetImage(msg))

    rospy.spin()

if __name__ == "__main__":
    listener()
