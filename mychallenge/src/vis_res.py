#!/usr/bin/env python

import rospy
import imghdr
import sensor_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import tf
import geometry_msgs.msg
import nav_msgs.msg
import visualization_msgs.msg
import PyKDL as kd
import math


heightmaps=[]
bridge = CvBridge()
pub = rospy.Publisher("/ObsOdom", nav_msgs.msg.Odometry, queue_size=10)
vis_pub = rospy.Publisher("/ObsVis",visualization_msgs.msg.Marker,queue_size=10)

CapPosX = [0]
CapPosY = [0]
CapPosZ = [0]
CapTwistX = [0]
CapTwistY = [0]
CapTwistZ = [0]

CapFrontPosX = [0]
CapFrontPosY = [0]
CapFrontPosZ = [0]
CapFrontTwistX = [0]
CapFrontTwistY = [0]
CapFrontTwistZ = [0]

RTKPosX = [0]
RTKPosY = [0]
RTKPosZ = [0]
RTKTwistX = [0]
RTKTwistY = [0]
RTKTwistZ = [0]



tracklet=np.genfromtxt("/home/xiaoyu/Documents/Tracklet.csv",delimiter=",")
def GetImage(msg,fmt='png'):
    img_stamp=msg.header.stamp.to_nsec()
    results = {}
    try:
        if hasattr(msg, 'format') and 'compressed' in msg.format:
            buf = np.ndarray(shape=(1, len(msg.data)),
                             dtype=np.uint8, buffer=msg.data)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            if cv_image.shape[2] != 3:
                print("Invalid image %s" % img_stamp)
                return results
            results['height'] = cv_image.shape[0]
            results['width'] = cv_image.shape[1]
            #Avoid re-encoding if we don't have to
            #cv2.imshow("image_heightmap", cv_image)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            #cv2.imshow("image_heightmap", cv_image)
    except CvBridgeError as e:
        print(e)
        
    #cv2.waitKey(5)
    front = kd.Vector(CapFrontPosX[0], CapFrontPosY[0], CapFrontPosZ[0])
    front2base = kd.Vector(0.4572, 0.0, 1.2192)
    rear = kd.Vector(CapPosX[0], CapPosY[0], CapPosZ[0])
    rear2base = kd.Vector(-0.8128,0,0.8636)
    velodyne2base = kd.Vector(1.5495,0.0,1.27)
    obs = kd.Vector(RTKPosX[0], RTKPosY[0], RTKPosZ[0] )
    xdirection = kd.Vector(1, 0, 0)
 
    #1
    #oblsgpslwh = kd.Vector(2.032, -0.7239, 1.6256)
    #obslwh = kd.Vector(4.2418, -1.4478, 1.5748)
    #2
    oblsgpslwh = kd.Vector(1.8288, -0.7874, 1.5748)
    obslwh = kd.Vector(4.191, -1.5748, 1.524)
    #3
    #oblsgpslwh = kd.Vector(1.9812, -0.8509, 1.4478)
    #obslwh = kd.Vector(4.5212, -1.7018, 1.397)

    minx=9999999999
    count=0
    for i in range(1,len(tracklet)):
        if np.absolute(tracklet[i][0]-img_stamp)<minx and tracklet[i][0]-img_stamp>=0:
            minx=np.absolute(tracklet[i][0]-img_stamp)
            count=i
    print(img_stamp)
    print(tracklet[count][0].astype(int))
    x=kd.Vector(tracklet[count][1], tracklet[count][2], tracklet[count][3])
    trans=kd.Vector(1.5495-4.5, 0, 1.27)
    x=x+trans
    odom = nav_msgs.msg.Odometry()
    odom.header.stamp = rospy.Time.now()
    odom.header.frame_id = "base_link"
    odom.pose.pose.position.x = x[0]
    odom.pose.pose.position.y = x[1]
    odom.pose.pose.position.z = x[2]
    #print("(%f,%f) (%f,%f) "%(CapPosX,CapPosY,RTKPosX[0],RTKPosY[0]))
    print("(%f,%f,%f) " % (x[0], x[1], x[2]))
    odom.twist.twist.linear.x = RTKTwistX[0] - CapTwistX[0]
    odom.twist.twist.linear.y = RTKTwistY[0] - CapTwistY[0]
    odom.twist.twist.linear.z = RTKTwistZ[0] - CapTwistZ[0]
    pub.publish(odom)

    marker=visualization_msgs.msg.Marker()
    marker.header.frame_id = "base_link";
    marker.header.stamp = msg.header.stamp
    marker.ns = "my_namespace";
    marker.id = 0;
    marker.type = visualization_msgs.msg.Marker.CUBE;
    marker.action = visualization_msgs.msg.Marker.MODIFY;
    marker.pose.position.x = x[0];
    marker.pose.position.y = x[1];
    marker.pose.position.z = x[2];
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = obslwh[0];
    marker.scale.y = obslwh[1];
    marker.scale.z = obslwh[2];
    marker.color.a = 1.0; # Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    vis_pub.publish( marker );



def listener():
    #cv2.namedWindow("image_heightmap")

    #cv2.namedWindow("image_raw")
    cv2.startWindowThread()
    rospy.init_node('mynode', anonymous=True)


    rospy.Subscriber('/image_raw',
                     sensor_msgs.msg.Image, lambda msg: GetImage(msg))

    rospy.spin()

if __name__ == "__main__":
    listener()
