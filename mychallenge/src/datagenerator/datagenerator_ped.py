# Copyright 2016 LIU XIAOYU. All Rights Reserved.
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
import imghdr
import sensor_msgs
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import tf
import sensor_msgs.point_cloud2 as pc2
import PyKDL as kd
import math
import cPickle
import pickle
import PyKDL as kd
import os
 
from PIL import Image
import threading
from std_msgs.msg import String
import random
from mychallenge.msg import MyOption

heightmaps = []
bridge = CvBridge()
time_stamp_count=[0]
tracklet = np.genfromtxt("/home/xiaoyu/Documents/Tracklet.csv", delimiter=",")
DEFAULT_FILE_DIR = '/home/xiaoyu/Documents/data/release/pedestrian./'
#DEFAULT_FILE_DIR = '/media/xiaoyu/B84E1E694E1E20A4/data/'
META_FILE_DIR = '/home/xiaoyu/Documents/data/release/pedestrian./training/'


class FeatureRecord:
    def __init__(self):
        self.features=list()
        #self.outpath='/media/xiaoyu/B84E1E694E1E20A4/data/'
        self.outpath='/home/xiaoyu/Documents/data/release/pedstrian_train/'
        self.tracklet = None
        self.dict = None
        self.lwh = None
        self.name = None
	self.count = 0
    def append(self, feature):
        self.features.append(feature)
    def clear(self):
        self.features=[]
        self.tracklet = None
        self.dict = None
        self.lwh = None
        self.name = None
	self.count = 0
    def dump(self):
        num = 0
        index = 0
        while num+500 < len(self.features):
            feature = self.features[num:num+500]
            output=open(self.outpath+self.name+'_'+str(index)+'.pkl',"wb")
            pickle.dump(feature,output)
            num += 500
            index += 1
        feature = self.features[num:]
        output=open(self.outpath+self.name+'_'+str(index)+'.pkl',"wb")
        pickle.dump(feature,output)
        file_obj = open(self.outpath+'info.txt','r+')
        file_obj.read()
        info = self.name + ' info:\n'
        info += '    csv num: ' + str(len(self.tracklet)) + '\n'
        info += '    features num: ' + str(len(self.features)) + '\n'
        file_obj.write(info)
        file_obj.close()
        
    def shuff(self):
        random.shuffle (self.features)
featurerecord=FeatureRecord()


def callback(data):
    commond = data.data
    if commond == "first":
        return
    elif commond == "write":
        print("writing to %s ..."%featurerecord.name)
        #featurerecord.shuff()
        featurerecord.dump()
        featurerecord.clear()
        print("write over")
    else:
        print("reading from %s..."%DEFAULT_FILE_DIR+commond)
        featurerecord.clear()
        file_name = None
        # for file in os.listdir(META_FILE_DIR+ commond):
        #     if os.path.splitext(file)[1]=='.bag':
        #         file_name = os.path.splitext(file)[0]
        trackletfile = DEFAULT_FILE_DIR + commond + '/Tracklet.csv'
        featurerecord.tracklet = np.genfromtxt(trackletfile, delimiter=",")
        featurerecord.dict = {}
        for idx,i in enumerate(featurerecord.tracklet):
            if idx == 0 :
                continue
            key = long(i[0])
            key /= 10000
            key *= 10000
            featurerecord.dict[key]=[i[1],i[2],i[3],i[6]]

        # metafile = META_FILE_DIR + commond + '/metadata.csv'
        # meta = np.genfromtxt(metafile, delimiter = ",")
        # featurerecord.lwh = [meta[1][2], meta[1][3], meta[1][4]]
        featurerecord.lwh = [1.708, 0.8, 0.8]
        featurerecord.name = 'data_'+commond
        print("bag has lidar frame: %f, car size: %f, %f, %f"%(len(featurerecord.tracklet)-1,
         featurerecord.lwh[0], featurerecord.lwh[1], featurerecord.lwh[2]))

def test(msg):
    featurerecord.count += 1
    img_stamp = msg.header.stamp.to_sec()
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
    img = np.dstack((height,intensity,ring,density))


    time_stamp = msg.header.stamp.to_nsec()
    print('%d  %d'%(featurerecord.count,time_stamp))
    time_stamp /= 10000
    time_stamp *= 10000
    #x,y,z,rz = featurerecord.dict[long(img_stamp)]
    minx = 1e7
    count = 0
    tracklet = featurerecord.tracklet
    for i in range(1, len(tracklet)):
        if np.absolute(tracklet[i][0] - time_stamp) < minx :#and tracklet[i][0] - time_stamp >= 0:
            minx = np.absolute(tracklet[i][0] - time_stamp)
            count = i
    x, y, z, rz = tracklet[count][1], tracklet[count][2], tracklet[count][3], tracklet[count][6]
    centPoint = np.array([x, y, z])
    print(centPoint)
    m = kd.Rotation.RPY(0,0,rz)



    obslwh = np.array(featurerecord.lwh)
    #obslwh += np.array([0.6, 0.6, 0.4])
    obslwh = obslwh / 2
    left_top = kd.Vector(obslwh[0], obslwh[1], obslwh[2])
    left_top = m * left_top 
    right_top = kd.Vector(obslwh[0], -obslwh[1], obslwh[2])
    right_top = m * right_top 
    left_bottom = kd.Vector(-obslwh[0], obslwh[1], obslwh[2])
    left_bottom = m * left_bottom 
    right_bottom = kd.Vector(-obslwh[0], -obslwh[1], obslwh[2])
    right_bottom = m * right_bottom
    l = np.max([left_top.x(), right_top.x(), left_bottom.x(), right_bottom.x()])
    w = np.max([left_top.y(), right_top.y(), left_bottom.y(), right_bottom.y()])
    obslwh = np.array([l, w, obslwh[2]])
    

    x = np.array([centPoint[1] - obslwh[1], centPoint[1] + obslwh[1]])
    y = np.array([centPoint[0] - obslwh[0], centPoint[0] + obslwh[0]])
    res = 0.1
    side_range = (-30,30)
    fwd_range = (-30,30)
    xp = (-x / res).astype(np.int32) 
    yp = (-y / res).astype(np.int32)  
    xp -= int(np.floor(side_range[0] / res))
    yp -= int(np.floor(fwd_range[0] / res))
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)   
    xp = np.clip(xp, 0, x_max)
    yp = np.clip(yp, 0, y_max)

    boundingbox=np.array([xp[1], yp[1], xp[0], yp[0], rz, 
        featurerecord.lwh[0], featurerecord.lwh[1], featurerecord.lwh[2], 
        centPoint[0], centPoint[1], centPoint[2], 
        obslwh[0], obslwh[1], obslwh[2]],dtype=np.float)

    bbox=boundingbox
    feature=dict()
    if bbox[3]-bbox[1]>5 and bbox[2]-bbox[0]>5:
        print(bbox[0:4])
        #====================================================
        #img feature contains 11 images: height_map, ring_map, density_map, and intensity slice 0~7 
        #====================================================
        feature["img"]=img
        #====================================================
        #gt_box feature contains bounding box ((x1,y1),(x2,y2)), physical car centroid (x,y,z), car scale (l, w, h)
        #====================================================
        feature["gt_box"]=bbox
        feature["label"]=1
        feature["orientation"]=rz/np.pi
    
        img_show=np.dstack((img[:,:,1],img[:,:,0],img[:,:,2]))
        img_show = img_show.copy()
        cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255))
        cv2.imshow("image",img_show)

        featurerecord.append(feature)
    else:
        print("bounding box too small")

def listener():
    #cv2.namedWindow("image_height")
    # cv2.namedWindow("image_ring")
    # cv2.namedWindow("image_density")
    cv2.namedWindow("image")
    cv2.startWindowThread()
    rospy.init_node('mynode', anonymous=True)


    rospy.Subscriber('/MyOption', String, callback)

    rospy.Subscriber('/feature', MyOption, test)
    rospy.spin()

if __name__ == "__main__":
    listener()
