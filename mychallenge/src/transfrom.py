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

pub = rospy.Publisher("/ObsOdom", nav_msgs.msg.Odometry, queue_size=10)
vis_pub = rospy.Publisher("/ObsVis",visualization_msgs.msg.Marker,queue_size=10)

CapPosX = 0
CapPosY = 0
CapPosZ = 0
CapTwistX = 0
CapTwistY = 0
CapTwistZ = 0

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


def get_yaw(p1, p2):
    if abs(p1[0] - p2[0]) < 1e-2:
        return 0.
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])


def GetObsRTK(msg):
    RTKPosX[0] = msg.pose.pose.position.x
    RTKPosY[0] = msg.pose.pose.position.y
    RTKPosZ[0] = msg.pose.pose.position.z
    RTKTwistX[0] = msg.twist.twist.linear.x
    RTKTwistY[0] = msg.twist.twist.linear.y
    RTKTwistZ[0] = msg.twist.twist.linear.z
    #print(RTKPosX)


def GetCapRTKFront(msg):
    CapFrontPosX[0] = msg.pose.pose.position.x
    CapFrontPosY[0] = msg.pose.pose.position.y
    CapFrontPosZ[0] = msg.pose.pose.position.z
    CapFrontTwistX[0] = msg.twist.twist.linear.x
    CapFrontTwistY[0] = msg.twist.twist.linear.y
    CapFrontTwistZ[0] = msg.twist.twist.linear.z




def GetCapRTKRear(msg):
    CapPosX = msg.pose.pose.position.x
    CapPosY = msg.pose.pose.position.y
    CapPosZ = msg.pose.pose.position.z
    CapTwistX = msg.twist.twist.linear.x
    CapTwistY = msg.twist.twist.linear.y
    CapTwistZ = msg.twist.twist.linear.z

    front = kd.Vector(CapFrontPosX[0], CapFrontPosY[0], CapFrontPosZ[0])
    front2base = kd.Vector(0.4572,0.0,1.2192)
    rear = kd.Vector(CapPosX, CapPosY, CapPosZ)
    rear2base = kd.Vector(-0.8128,0,0.8636)
    velodyne2base = kd.Vector(1.5495,0.0,1.27)
    obs = kd.Vector(RTKPosX[0], RTKPosY[0], RTKPosZ[0] )
    xdirection = kd.Vector(1, 0, 0)
    yaw = get_yaw(front, rear)
    rot_z = kd.Rotation.RotZ(-yaw)
    oblsgpslwh = kd.Vector(2.032,0.7239,1.6256)
    obslwh = kd.Vector(4.2418, 1.4478, 1.5748)
    res = rot_z * (obs-front)
    # odom.pose.pose.position.x=RTKPosX[0]-CapPosX;
    # odom.pose.pose.position.y=RTKPosY[0]-CapPosY;
    # odom.pose.pose.position.z=RTKPosZ[0]-CapPosZ;
    x = res - front2base + velodyne2base# - oblsgpslwh + obslwh/2 

    odom = nav_msgs.msg.Odometry()
    odom.header.stamp = msg.header.stamp
    odom.header.frame_id = "base_link"
    odom.pose.pose.position.x = x[0]
    odom.pose.pose.position.y = x[1]
    odom.pose.pose.position.z = x[2]
    #print("(%f,%f) (%f,%f) "%(CapPosX,CapPosY,RTKPosX[0],RTKPosY[0]))
    print("(%f,%f,%f) " % (x[0], x[1], x[2]))
    odom.twist.twist.linear.x = RTKTwistX[0] - CapTwistX
    odom.twist.twist.linear.y = RTKTwistY[0] - CapTwistY
    odom.twist.twist.linear.z = RTKTwistZ[0] - CapTwistZ
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
    #only if using a MESH_RESOURCE marker type:
    #marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
    vis_pub.publish( marker );


def listener():
    print("Transfrom start")
    rospy.init_node('CapTranObs', anonymous=True)

    rospy.Subscriber('/objects/obs1/rear/gps/rtkfix',
                     nav_msgs.msg.Odometry, lambda msg: GetObsRTK(msg))
    rospy.Subscriber('/objects/capture_vehicle/rear/gps/rtkfix',
                     nav_msgs.msg.Odometry, lambda msg: GetCapRTKRear(msg))
    rospy.Subscriber('/objects/capture_vehicle/front/gps/rtkfix',
                     nav_msgs.msg.Odometry, lambda msg: GetCapRTKFront(msg))

    rospy.spin()

if __name__ == "__main__":
    listener()
