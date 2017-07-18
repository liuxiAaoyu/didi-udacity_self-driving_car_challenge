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

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.



tl = np.array([1.94908961,  0.14588651,  0.30824467])
tr = np.array([2.46187825, -0.62988707,  0.13369188])
bl = np.array([1.84270041,  0.28992585, -0.62694901])
br = np.array([2.35213666, -0.48750511, -0.79589406])


def distance(p1, p2):
    return np.sqrt((p1 - p2).dot(p1 - p2))


def invec(p1, p2):
    v = p2 - p1
    v = v / distance(p1, p2)
    return v
p1 = tl + 0.116 * invec(tl, tr)
p2 = tr + 0.125 * invec(tr, tl)
p3 = bl + 0.116 * invec(bl, br)
p4 = br + 0.125 * invec(br, bl)


v1 = invec(tl, tr)
v2 = invec(bl, br)

v = invec(p1, p3)

start = p1
end = p1 + 0.7 * v1
start += 0.02 * v
end += 0.02 * v
# for i in range(10):

#     x1=np.linspace(start[0],end[0],100)
#     y1=np.linspace(start[1],end[1],100)
#     z1=np.linspace(start[2],end[2],100)
#     start+=0.1*v
#     end+=0.1*v
#     for i in range(len(x1)):
#         print("%f %f %f 255 0 0"%(x1[i],y1[i],z1[i]))

s = start + 0.8 * v
point3d = list()
for i in range(6):
    s += 0.1 * v1
    s1 = 0
    s1 += s
    for i in range(8):
        s2=0
        s2+=s1
        point3d.append(s2)
        #print("%f %f %f 0 0 255" % (s1[0], s1[1], s1[2]))
        s1 += -0.1 * v
A = np.zeros((48 * 2, 12), dtype=np.float64)
O = np.zeros((48 * 2, 1), dtype=np.float)
P = np.zeros((12, 1), dtype=np.float)
# for i in range(48):
#     x = point3d[i][0]
#     y = point3d[i][1]
#     z = point3d[i][2]
#     objp[i]=np.array([x,y,z])
#     u = imgpoints[0][i][0][0]
#     v = imgpoints[0][i][0][1]
#     print("%f %f %f %f %f " % (x,y,z,u,v))
    # tempa0 = np.array([x, y, z, 1, 0, 0, 0, 0, u * x,
    #                    u * y, u * z, u], dtype=np.float64)
    # tempa1 = np.array([0, 0, 0, 0, x, y, z, 1, v * x,
    #                    v * y, v * z, v], dtype=np.float64)
    # A[i * 2] = tempa0
    # A[i * 2 + 1] = tempa1


# Make a list of calibration images
images = list()
images.append(
    "/home/xiaoyu/Documents/1DIDIUDA/tf_testprocess/calbration/1492642061958477906.jpg")
img = cv2.imread("/home/xiaoyu/Documents/1DIDIUDA/tf_testprocess/calbration/1492642061958477906.jpg")
img_size = (img.shape[1], img.shape[0])

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8, 6), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)

cv2.destroyAllWindows()

for i in range(48):
    x = point3d[i][0]
    y = point3d[i][1]
    z = point3d[i][2]
    objp[i]=np.array([x,y,z])
    u = imgpoints[0][i][0][0]
    v = imgpoints[0][i][0][1]
    print("%f %f %f %f %f " % (x,y,z,u,v))

X = np.dot(A.T, A)

# for j in range(12):
#     for i in range(12):
#         print("%.5f" % (X[j][i])),
#     print(" ")

# W, V = np.linalg.eig(X)
# for j in range(12):
#     print(j)
#     print(W[j])
#     Xa = np.dot(X, V[:][j])
#     Xb = W[j] * V[:][j]

#     for i in range(12):
#         print("%f %f %f" % (V[i][j], Xa[i], Xb[i]))
# for i in range(12):
#     if (np.dot(X, V[:][i]) == W[i] * V[:][i]).all():
#         print 'true'
#     else:
#         print 'false'



# vv = np.array([-0.142235947464232, -0.0177089393813047, -0.339198649405639, 0.377438520958456, 0.108986144643184, -0.0119585894145019,
#                0.373296657462745, -0.318133927755863, 0.256048420386704, 0.118156454150779, 0.227072139321146, -0.581614728243648])
# vv = np.array([-0.0905097731808156,-0.0978288424227322,0.168973189334219,0.142016537600785,0.169854011119208,0.150962957504195,-0.171880704770830,-0.303548565251203,-0.211128387310856,-0.308343489733161,0.750148281966363,0.240454999088022])

# print(np.dot(X, vv.T))
# #
# P=np.array([[vv[0], vv[1], vv[2], vv[3]], [vv[4], vv[5],
#                vv[6], vv[7]], [vv[8], vv[9], vv[10], vv[11] ]])
# for i in range(48):
#     point=np.array([point3d[i][0],point3d[i][1],point3d[i][2],1])

#     point=np.dot(P,point)
#     print(point,point[0]/point[2],point[1]/point[2])


    # Do camera calibration given object points and image points


#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
cameraMatrix = np.array([[1384.621562, 0.000000, 625.888005],
                         [0.000000, 1393.652271, 559.626310],
                         [0.000000, 0.000000, 1.000000]])
distcoeffs = np.array([-0.152089, 0.270168, 0.003143, -0.005640, 0.000000])  
distcoeffs = np.array([0,0,0,0, 0.000000])              
rvecs=None
tvecs = None
ret, rvecs, tvecs = cv2.solvePnP(objp, imgpoints[0],cameraMatrix, distcoeffs)

#rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, imgpoints[0], cameraMatrix, distcoeffs)


print(rvecs)
print(tvecs)
print(ret)

#cv2.solvePnP2(objp, imgpoints[0],cameraMatrix, distcoeffs,rvecs, tvecs)
imgp,jac=cv2.projectPoints(objp, rvecs, tvecs, cameraMatrix, distcoeffs)
print(imgp)
print(jac)
_t=np.array([[23,-4,-0.57],[-2,-0.07,-0.57]])
imgp,jac=cv2.projectPoints(_t, rvecs, tvecs, cameraMatrix, distcoeffs)
print(imgp)