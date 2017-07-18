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

import tensorflow as tf
import pickle
import numpy as np
import random
import sys
import os
####http://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow
FLAGS = tf.app.flags.FLAGS

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _EncodedFloatFeature( ndarray):
    return tf.train.Feature(float_list=tf.train.FloatList(
        value=ndarray.flatten().tolist()))

def _EncodedInt64Feature( ndarray):
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=ndarray.flatten().tolist()))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


input_image = tf.placeholder(dtype = tf.uint8)
png_encode = tf.image.encode_png(input_image)


def ndarray2png_fn(sess, image):
    image_data = sess.run(png_encode,feed_dict={input_image : image})
    return image_data


def process_data(data, tfrecord_writer, height, width, channle, sess):
    image_data = data['img']
    image_data = image_data[:,:,:3]
    #image_raw = image_data.tostring()
    image_png = ndarray2png_fn(sess, image_data)
    gt_box = data['gt_box'][:4]
    label = data['label']
    image_format = b'PNG'
    xmin = gt_box[0] / width
    xmax = gt_box[2] / width
    ymin = gt_box[1] / height
    ymax = gt_box[3] / height

    ##############################################################################
    # centPoint = data['gt_box'][8:10]
    # x = np.array([centPoint[1] - 0.4, centPoint[1] + 0.4])
    # y = np.array([centPoint[0] - 0.4, centPoint[0] + 0.4])
    # res = 0.1 
    # side_range = (-30,30)
    # fwd_range = (-30,30)
    # xp = (-x / res).astype(np.int32) 
    # yp = (-y / res).astype(np.int32)  
    # xp -= int(np.floor(side_range[0] / res))
    # yp -= int(np.floor(fwd_range[0] / res))
    # x_max = int((side_range[1] - side_range[0]) / res)
    # y_max = int((fwd_range[1] - fwd_range[0]) / res)   
    # xp = np.clip(xp, 0, x_max)
    # yp = np.clip(yp, 0, y_max)

    # xmin = xp[1] / width
    # xmax = xp[0] / width
    # ymin = yp[1] / height
    # ymax = yp[0] / height
    ##############################################################################


    example = tf.train.Example(features = tf.train.Features(
        feature = {
            'image/encoded' : bytes_feature(image_png),
            'image/format' : bytes_feature(image_format),
            'image/height': int64_feature(height),
            'image/width' : int64_feature(width),
            'image/channels' : int64_feature(channle),
            'image/object/bbox/xmin' : float_feature(xmin),
            'image/object/bbox/xmax' : float_feature(xmax),
            'image/object/bbox/ymin' : float_feature(ymin),
            'image/object/bbox/ymax' : float_feature(ymax),
            'image/object/bbox/label' : int64_feature(label),
            'image/orientation' : float_feature(data['orientation']),
        }
    ))
    tfrecord_writer.write(example.SerializeToString())



PERFIX = "/home/xiaoyu/Documents/data/release/tfrecord/data.tfrecord"
count = []
count.append(0)


def processfile(filename, tfrecord_writer, sess):
    data = list()
    _tdata = pickle.load(open(filename,"rb"), encoding = "iso-8859-1")
    random.shuffle(_tdata)
    data.append(_tdata)
    height = data[0][0]["img"].shape[0]
    width = data[0][0]["img"].shape[1]
    channle = data[0][0]["img"].shape[2]
    i = 0
    fidx = 0
    while i < len(data):
        j = 0
        while j < len(data[i]):
            sys.stdout.write('\rconverting image %d/%d'%(j+1,len(data[i])))
            sys.stdout.flush()
            process_data(data[i][j], tfrecord_writer, height, width, channle, sess)
            j += 1
            count[0] += 1
        i += 1

def listdir(path, list_name, path_name):  
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name, path_name)  
        elif os.path.splitext(file_path)[1]=='.pkl':  
            list_name.append(file_path)
            path_name.append(os.path.split(file_path)[0])

def main(_):
    
    filelist = []
    pathlist = []
    listdir('/home/xiaoyu/Documents/data/release/car_train', filelist, pathlist)     
    filelist2 = []
    pathlist2 = []
    listdir('/home/xiaoyu/Documents/data/release/pedstrian_train', filelist2, pathlist2) 
    filelist.extend(filelist2)



    random.shuffle(filelist)
    for i in filelist:
        print(i)

    num=0
    fileidx=0
    with tf.Session() as sess:
        tf_filename = PERFIX
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            
            while fileidx < len(filelist):
                data = []
                while len(data) < 2000 and fileidx < len(filelist):
                    fl = filelist[fileidx]
                    tdata = pickle.load(open(fl,'rb'))#, encoding = "iso-8859-1")
                    fileidx += 1
                    if os.path.split(os.path.split(fl)[0])[1] == 'pedstrian_train':
                        for i in range(len(tdata)):
                            tdata[i]['label'] = 2
                    data.extend(tdata)

                
                random.shuffle(data)
            
                #tf_filename = PERFIX + str(num) + '.tfrecord'
                num += 1
                height = data[0]["img"].shape[0]
                width = data[0]["img"].shape[1]
                channle = data[0]["img"].shape[2]
            
                for j in data:
                    process_data(j, tfrecord_writer, height, width, channle, sess)
                    count[0] += 1
                    sys.stdout.write('\rconverting image %d'%(count[0]))
                    sys.stdout.flush()
                

        # idx = range(0,len(filelist),3)
        # idx.append(len(filelist))
        # for i in range(idx-1):
        #     fl = filelist[idx[i]:idx[i+1]]
        #     data = []
        #     for tf in fl:
        #         tdata = pickle.load(open(tf),'rb')
        #         data.extend(tdata)
        #     random.shuffle(data)
            
        #     tf_filename = PERFIX + str(num) + '.tfrecord'
        #     num += 1
        #     height = data[0]["img"].shape[0]
        #     width = data[0]["img"].shape[1]
        #     channle = data[0]["img"].shape[2]
        #     with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        #         for j in data:
        #             process_data(j, tfrecord_writer, height, width, channle, sess)
        #             count[0] += 1
        #             sys.stdout.write('\rconverting image %d'%(count[0]))
        #             sys.stdout.flush()
    print('\n total: %d '%count[0])
    print('Finished')


if __name__ == "__main__":
    tf.app.run()
