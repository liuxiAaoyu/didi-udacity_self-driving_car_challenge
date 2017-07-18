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

from PIL import Image
import threading
from std_msgs.msg import String
import random
from mychallenge.msg import MyOption

heightmaps = []
bridge = CvBridge()
time_stamp_count=[0]
tracklet = np.genfromtxt("/home/xiaoyu/Documents/Tracklet.csv", delimiter=",")
DEFAULT_FILE_DIR = '/home/xiaoyu/Documents/data/release/train'

#blog.csdn.net/hehe123456zxc/article/details/52264829
#www.2cto.com/kf/201504/395335.html
class HrdSemaphore:
    def __init__(self):
        self.hrd_img = None
        self.stamp = None
        self.event = threading.Event()
        self.event.clear()
    def set(self, im, stamp):
        self.hrd_img = im
        self.stamp = stamp
        self.event.set()
    def get(self):
        self.event.wait()
        return self.hrd_img, self.stamp
    def clear(self):
        self.event.clear()

hrd_img=HrdSemaphore()

class FeatureRecord:
    def __init__(self):
        self.features=list()
        self.outpath="/home/xiaoyu/Documents/data/Didi-Training/mydata/"
        self.tracklet = None
        self.name = None
    def append(self, feature):
        self.features.append(feature)
    def clear(self):
        self.features=[]
        self.tracklet = None
        self.name = None
    def dump(self):
        output=open(self.outpath+self.name,"wb")
        pickle.dump(self.features,output,2)
    def shuff500(self):
        while len(self.features)<500:
            print(len(self.features))
            self.features.extend(self.features)
        self.features = self.features[:500]
        random.shuffle (self.features)
featurerecord=FeatureRecord()


def GetHeightMap(msg, fmt='png'):
    print(rospy.get_time())
    img_stamp = msg.header.stamp.to_nsec()
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

    #cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    hrd_img.set(cv_image, img_stamp)



def GetImage(msg, fmt='png'):
    img_stamp = msg.header.stamp.to_nsec()
    name="/home/xiaoyu/Downloads/calibration/1./"+str(img_stamp)+".jpg"
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("image", cv2.pyrDown(cv_image))
        cv2.imwrite(name,cv_image)
    except CvBridgeError as e:
        print(e)




def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)



def birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10, 10),
                          res=0.1,
                          min_height=-2.73,
                          max_height=1.27,
                          time_stamp=None,
                          saveto=None):
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices] / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_lidar[indices] / res).astype(np.int32)  # y axis is -x in LIDAR
    # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0] / res))
    y_img -= int(np.floor(fwd_range[0] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_values 

    #cv2.imshow("image",im)
    # Convert from numpy array to a PIL image
    #im = Image.fromarray(im)
    #im.save("/home/xiaoyu/a.png")
    # SAVE THE IMAGE
    #if saveto is not None:
    #    im.save(saveto)
    # minx = 9999999999
    # count = 0
    # for i in range(1, len(tracklet-12)):
    #     if np.absolute(tracklet[i][0] - time_stamp) < minx and tracklet[i][0] - time_stamp >= 0:
    #         minx = np.absolute(tracklet[i][0] - time_stamp)
    #         count = i+12
    # centPoint = np.array([tracklet[count][1], tracklet[
    #                      count][2], tracklet[count][3]])
    # obslwh = np.array([4.5212, 1.7018, 1.397])
    # obslwh += np.array([0.6, 0.8, 0.4])
    # obslwh = obslwh / 2

    # x = np.array([centPoint[1] - obslwh[1], centPoint[1] + obslwh[1]])
    # y = np.array([centPoint[0] - obslwh[0], centPoint[0] + obslwh[0]])
    # xp = (-x / res).astype(np.int32)  # x axis is -y in LIDAR
    # yp = (-y / res).astype(np.int32)  # y axis is -x in LIDAR

    # xp -= int(np.floor(side_range[0] / res))
    # yp -= int(np.floor(fwd_range[0] / res))
    # boundingbox = np.zeros([y_max, x_max], dtype=np.uint8)
    # xp = np.clip(xp, 0, x_max)
    # yp = np.clip(yp, 0, y_max)
    # cv2.rectangle(boundingbox, (xp[0], yp[0]), (xp[1], yp[1]), (255))

    # out = np.dstack((boundingbox, boundingbox, im))
    # cv2.imshow("image", out)
    return  im

def birds_eye_height_slices(points,
                            n_slices=8,
                            height_range=(-2.73, 1.27),
                            side_range=(-10, 10),
                            fwd_range=(-10, 10),
                            res=0.1,
                            time_stamp=None,
                            ):
    """ Creates an array that is a birds eye view representation of the
        reflectance values in the point cloud data, separated into different
        height slices.

    Args:
        points:     (numpy array)
                    Nx4 array of the points cloud data.
                    N rows of points. Each point represented as 4 values,
                    x,y,z, reflectance
        n_slices :  (int)
                    Number of height slices to use.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the sensor.
                    The slices calculated will be within this range, plus
                    two additional slices for clipping all values below the
                    min, and all values above the max.
                    Default is set to (-2.73, 1.27), which corresponds to a
                    range of -1m to 3m above a flat road surface given the
                    configuration of the sensor in the Kitti dataset.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    Left and right limits of rectangle to look at.
                    Defaults to 10m on either side of the car.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
                    Defaults to 10m behind and 10m in front.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size along the front and side plane.
    """
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    i_points = points[:, 3]  # Reflectance

    # FILTER INDICES - of only the points within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    ss = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    # KEEPERS - The actual points that are within the desired  rectangle
    y_points = y_points[indices]
    x_points = x_points[indices]
    z_points = z_points[indices]
    i_points = i_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_points / res).astype(np.int32)  # y axis is -x in LIDAR
                                               # direction to be inverted later
    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0] / res))
    y_img -= int(np.floor(fwd_range[0] / res))

    # ASSIGN EACH POINT TO A HEIGHT SLICE
    # n_slices-1 is used because values above max_height get assigned to an
    # extra index when we call np.digitize().
    bins = np.linspace(height_range[0], height_range[1], num=n_slices-1)
    slice_indices = np.digitize(z_points, bins=bins, right=False)

    # RESCALE THE REFLECTANCE VALUES - to be between the range 0-255
    pixel_values = scale_to_255(i_points, min=0.0, max=255.0)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    # -y is used because images start from top left
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max, n_slices], dtype=np.uint8)
    im[-y_img, x_img, slice_indices] = pixel_values
    #for i in range(0,8):
    #cv2.imshow("image_raw %d"%(0), cv2.pyrDown(im[:,:,0]))
    
    minx = 9999999999
    count = 0
    tracklet = featurerecord.tracklet
    
    for i in range(1, len(tracklet)):
        if np.absolute(tracklet[i][0] - time_stamp) < minx :#and tracklet[i][0] - time_stamp >= 0:
            minx = np.absolute(tracklet[i][0] - time_stamp)
            count = i
    tracklet=np.vstack((tracklet[:12],tracklet))
    if count <12:
        count =1
    #tracklet =  tracklet[12:]
    centPoint = np.array([tracklet[count][1], tracklet[
                         count][2], tracklet[count][3]])
    print(centPoint)
    #obslwh = np.array([4.2418, 1.4478, 1.5748])
    #obslwh = np.array([4.191, 1.5748, 1.524])
    obslwh = np.array([4.5212, 1.7018, 1.397])
    obslwh += np.array([1, 0.8, 0.4])
    obslwh = obslwh / 2

    x = np.array([centPoint[1] - obslwh[1], centPoint[1] + obslwh[1]])
    y = np.array([centPoint[0] - obslwh[0], centPoint[0] + obslwh[0]])
    xp = (-x / res).astype(np.int32)  # x axis is -y in LIDAR
    yp = (-y / res).astype(np.int32)  # y axis is -x in LIDAR

    xp -= int(np.floor(side_range[0] / res))
    yp -= int(np.floor(fwd_range[0] / res))
    
    xp = np.clip(xp, 0, x_max)
    yp = np.clip(yp, 0, y_max)

    boundingbox=np.array([xp[1], yp[1], xp[0], yp[0], centPoint[0], 
        centPoint[1], centPoint[2], obslwh[0], obslwh[1], obslwh[2]],dtype=np.float)

    return im,boundingbox



def GetVelodynePoints(msg):
    img_stamp = msg.header.stamp.to_nsec()
    time_stamp_count[0]+=1
    print(time_stamp_count[0],img_stamp)
    _t = rospy.get_time()
    lidar = pc2.read_points(msg,skip_nans=True)
    print(type(lidar))
    lidar = np.array(list(lidar))
    _t2 = rospy.get_time()
    output = open('/home/xiaoyu/ped.pkl', 'wb')
    pickle.dump(lidar,output)
    # im,bbox = birds_eye_height_slices(lidar,
    #                     n_slices=8,
    #                     height_range=(-2.0, 0.27),
    #                     side_range=(-30, 30),
    #                     fwd_range=(-30, 30),
    #                     res=0.1,
    #                     time_stamp=img_stamp)
    im = birds_eye_point_cloud(lidar,
                          side_range=(-30, 30),
                          fwd_range=(-30, 30),
                          res=0.1,
                          min_height=-2.73,
                          max_height=1.27)
    print('slice time: %s'%(_t2-_t))
    #hrd,stamp=hrd_img.get()
    _t3 = rospy.get_time()
    print('feature time: %s'%(_t3-_t2))
    hrd_img.clear()
    feature=dict()
    bbox=[0,0,0,0]
    if bbox[3]-bbox[1]>30 :
        print(bbox[0:4])
        #====================================================
        #img feature contains 11 images: height_map, ring_map, density_map, and intensity slice 0~7 
        #====================================================
        feature["img"]=np.array(np.dstack((hrd,im))) 
        #====================================================
        #gt_box feature contains bounding box ((x1,y1),(x2,y2)), physical car centroid (x,y,z), car scale (l, w, h)
        #====================================================
        feature["gt_box"]=bbox
    
        img_show=feature["img"][:,:,0]
        img_show = img_show.copy()
        cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255))
        cv2.imshow("image",img_show)

        featurerecord.append(feature)
    else:
        print("bounding box too small")
    # if time_stamp_count[0]%659==0:
    #     print("writing...")
    #     featurerecord.dump("data.pkl")
    #     featurerecord.clear()
    #     print("write over")

def callback(data):
    commond = data.data
    if commond == "first":
        return
    elif commond == "write":
        print("writing to %s ..."%featurerecord.name)
        featurerecord.shuff500()
        featurerecord.dump()
        featurerecord.clear()
        print("write over")
    else:
        print("reading from %s"%DEFAULT_FILE_DIR+commond)
        featurerecord.clear()
        trackletfile = DEFAULT_FILE_DIR + commond + '/Tracklet.csv'
        featurerecord.tracklet = np.genfromtxt(trackletfile, delimiter=",")
        featurerecord.name = 'data_'+commond[:1]+'_'+commond[3:]+'.pkl'

def test(msg):
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
    img = np.dstack((intensity,height,ring))
    _t = rospy.get_time()
    print(_t-img_stamp)
    img_show = img.copy()
    #cv2.imwrite('/home/xiaoyu/t.png',img_show)
    cv2.imshow("image_height",img)

def listener():
    cv2.namedWindow("image_height")
    # cv2.namedWindow("image_ring")
    # cv2.namedWindow("image_density")
    # cv2.namedWindow("image")
    cv2.startWindowThread()
    rospy.init_node('mynode', anonymous=True)

    # rospy.Subscriber('/heightmap/pointcloud',
    #                  sensor_msgs.msg.Image, lambda msg: GetHeightMap(msg))
    # #rospy.Subscriber('/image_raw',
    # #                 sensor_msgs.msg.Image, lambda msg: GetImage(msg))

    rospy.Subscriber('/velodyne_points',
                     sensor_msgs.msg.PointCloud2, lambda msg: GetVelodynePoints(msg))

    # rospy.Subscriber('/MyOption', String, callback)

    rospy.Subscriber('/feature', MyOption, test)
    rospy.spin()

if __name__ == "__main__":
    listener()
