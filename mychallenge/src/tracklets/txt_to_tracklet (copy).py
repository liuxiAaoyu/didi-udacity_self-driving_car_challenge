#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Bag Processing
"""

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import cv2
import math
import imghdr
import argparse
import functools
import matplotlib
import numpy as np
import pandas as pd

from bag_topic_def import *
from bag_utils import *
from generate_tracklet import *




def interpolate_to_camera(camera_df, other_dfs, filter_cols=[]):
    if not isinstance(other_dfs, list):
        other_dfs = [other_dfs]
    if not isinstance(camera_df.index, pd.DatetimeIndex):
        print('Error: Camera dataframe needs to be indexed by timestamp for interpolation')
        return pd.DataFrame()

    for o in other_dfs:
        o['timestamp'] = pd.to_datetime(o['timestamp'])
        o.set_index(['timestamp'], inplace=True)
        o.index.rename('index', inplace=True)

    merged = functools.reduce(lambda left, right: pd.merge(
        left, right, how='outer', left_index=True, right_index=True), [camera_df] + other_dfs)
    merged.interpolate(method='time', inplace=True,
                       limit=100, limit_direction='both')

    filtered = merged.loc[camera_df.index]  # back to only camera rows
    filtered.fillna(0.0, inplace=True)
    filtered['timestamp'] = filtered.index.astype(
        'int')  # add back original timestamp integer col
    if filter_cols:
        if not 'timestamp' in filter_cols:
            filter_cols += ['timestamp']
        filtered = filtered[filter_cols]

    return filtered

class obsitems(object):
    def __init__(self, id):
        self.id = id;
        self.arr = defaultdict(list)
        self.l = 0
        self.w = 0
        self.h = 0
    
    def append(self, timestamp, tx, ty, tz, rz, l , w, h):
        self.arr["timestamp"].append(timestamp)
        self.arr["tx"].append(tx)  
        self.arr["ty"].append(ty)
        self.arr["tz"].append(tz) 
        self.arr["rx"].append(0)
        self.arr["ry"].append(0)
        self.arr["rz"].append(rz)
        self.l = l
        self.w = w
        self.h = h

    def convt2pd(self):
        self.arr_df = pd.DataFrame(data=self.arr, columns=["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"])

    def interp(self, camera_index_df):
        self.obstacle_interp = interpolate_to_camera(
            camera_index_df, self.arr_df, filter_cols=["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"])
    
    def generate_tracklet(self, obsname='Pedestrian'):
        xx = self.obstacle_interp.to_dict(orient='record')
        firstframe = self.arr["timestamp"][0]-2e8
        firstindex = 0;
        while xx[firstindex]['timestamp']<firstframe:
            firstindex += 1
        length = len(xx) -1
        lastframe = self.arr["timestamp"][len(self.arr["timestamp"])-1]
        lastindex = length
        while xx[lastindex]['timestamp']>lastframe:
            lastindex -= 1
        
        self.obs_tracklet = Tracklet(
            object_type=obsname, l=self.l, w=self.w, h=self.h, first_frame=firstindex)

        for i in range(firstindex, lastindex):
            ds = dict()
            ds['tx'] = xx[i]['tx']
            ds['ty'] = xx[i]['ty']
            ds['tz'] = xx[i]['tz']
            ds['rx'] = xx[i]['rx']
            ds['ry'] = xx[i]['ry']
            ds['rz'] = xx[i]['rz']
            self.obs_tracklet.poses.append(ds)

def listdir(path, list_name):  
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        elif os.path.splitext(file_path)[1]=='.csv':  
            list_name.append(file_path)



def main():
    parser = argparse.ArgumentParser(
        description='Convert txt and csv to tacklet.xml.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/home/xiaoyu/results/_n/',
                        help='Output folder')
    parser.add_argument('-cfd', '--camera_file_dir', type=str, nargs='?', default='/home/xiaoyu/Documents/data/release/car./',
                        help='camera file')
    parser.add_argument('-cid', '--camera_index_dir', type=str, nargs='?', default='/home/xiaoyu/catkin_ws/src/mychallenge/src/tracklets/cameraIndex/',
                        help='camera file')
    parser.add_argument('-cfn', '--camera_file_name', type=str, nargs='?', default='ford07',
                        help='camera file')
    parser.add_argument('-if', '--inputfile', type=str, nargs='?', default='/home/xiaoyu/results/_n/',
                        help='lidar point file')


    args = parser.parse_args()
    
    dataset_outdir = args.outdir

    camerafile = args.camera_file_dir + args.camera_file_name + '/cap_camera.csv'
    
    cvslist = []
    listdir(args.camera_index_dir, cvslist)

    txtpath = args.inputfile + args.camera_file_name + '.txt'


    # For bag sets that may have missing metadata.csv file
    default_metadata_car = [{
        'obstacle_name': 'obs1',
        'object_type': 'Car',
    }]
    default_metadata_ped = [{
        'obstacle_name': 'obs1',
        'object_type': 'Pedestrian',
        'l': 0.8,
        'w': 0.8,
        'h': 1.708,
    }]

    rtk_cols = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]
    camera_cols = ["timestamp", "width", "height", "frame_id", "filename"]


    camera_df = pd.DataFrame()
    camera_df = pd.read_csv(camerafile, header=None, names=camera_cols)


    fp = open(txtpath)
    tempar = list()
    items = dict()
    for lines in fp.readlines():
        lines = lines.replace("\n", "").split(" ")
        td = dict()
        td["id"] = (int(lines[0]))
        td["timestamp"] = (long(lines[1]))
        td["tx"] = (float(lines[2]))
        td["ty"] = (float(lines[3]))
        td["tz"] = (float(lines[4]))
        rrz = float(lines[5])
        td["rz"] = (rrz)
        td["l"] = (float(lines[7]))
        td["w"] = (float(lines[8]))
        td["h"] = (float(lines[9]))

        if td["id"] in items.keys():
            items[td["id"]].append(td["timestamp"], td["tx"], td["ty"], td["tz"], td["rz"], td["l"], td["w"], td["h"]);
        else:
            _tempitem= obsitems(td["id"])
            _tempitem.append(td["timestamp"], td["tx"], td["ty"], td["tz"], td["rz"], td["l"], td["w"], td["h"]);
            items[td["id"]] = _tempitem

        tempar.append(td)
    fp.close()
    #tempar.sort(key=lambda k: (k.get('timestamp', 0)))

    camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
    camera_df.set_index(['timestamp'], inplace=True)
    camera_df.index.rename('index', inplace=True)

    camera_index_df = pd.DataFrame(index=camera_df.index)
    
    collection = TrackletCollection()

    if args.camera_file_name == 'ped_test':
        obsname = 'Pedestrian'
    else:
        obsname = 'Car'

    for (keys, item) in items.items(): 
        item.convt2pd();
        item.interp(camera_index_df)
        item.generate_tracklet(obsname)
        collection.tracklets.append(item.obs_tracklet)


    tracklet_path = os.path.join(dataset_outdir, args.camera_file_name+'.xml')
    collection.write_xml(tracklet_path)
    print("Finish")



if __name__ == '__main__':
    main()
