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
##########################################################################################
#                              top1                 top2
#                        ......+...................+...........
#                        .                                    .
#                        + fetf1                              +right1
#                        .                                    .
#                        .                                    .
#                        + left2                              +right2
#                        ......+.........................+.....
#                               bottom1                  bottom2          
##########################################################################################
top2 = np.array([2.36102, -0.477303, 0.168024])
top1 = np.array([2.01676, 0.043513, 0.28521])

left1 = np.array([1.94545, 0.15057, 0.27571])
left2 = np.array([1.87127, 0.246737, -0.356184])

right1 = np.array([2.46006,-0.627528,0.11829])
right2 = np.array([2.3986,-0.547788,-0.402318])

bottom1 = np.array([1.905020, 0.188814, -0.64848])
bottom2 = np.array([2.17904, -0.255776, -0.743152])




t2t1 = top2 - top1
l1t1 = left1 - top1
l2t1 = left2 - top1
print(t2t1)
print(l1t1)
print(l2t1)

a = t2t1[0] * l1t1[1] * l2t1[2] + l1t1[0] * l2t1[1] * t2t1[2] + l2t1[0] * t2t1[1] * l1t1[2] - \
    l2t1[0] * l1t1[1] * t2t1[2] - t2t1[0] * \
    l2t1[1] * l1t1[2] - l1t1[0] * t2t1[1] * l2t1[2]
print(a)


def cal(l11,l12,l21,l22):

    a = [[l11[0]-l12[0],l22[0]-l21[0]],[l11[1]-l12[1],l22[1]-l21[1]],[l11[2]-l12[2],l22[2]-l21[2]]]
    A = np.array(a)

    b = np.array([l22[0]-l12[0], l22[1]-l12[1], l22[2]-l12[2]])

    c,resid,rank,sigma = np.linalg.lstsq(A, b)

    print("solution:")
    print(c)

    p1=(l11-l12)*c[0]+l12
    print("point :")
    print(p1)
    p2=(l21-l22)*c[1]+l22
    print(p2)

print("top left")
cal(top1,top2,left1,left2)

print("top right")
cal(top1,top2,right1,right2)

print("bottom left")
cal(bottom1,bottom2,left1,left2)

print("bottom right")
cal(bottom1,bottom2,right1,right2)



# bottom right 2.35213666 -0.48750511 -0.79589406 255 0 0 
# bottom left 1.84270041  0.28992585 -0.62694901 0 255 0
# top right 2.46187825 -0.62988707  0.13369188 255 0 0
# top lefet 1.94908961  0.14588651  0.30824467 0 255 0

tl=np.array([1.94908961,  0.14588651,  0.30824467])
tr=np.array([2.46187825, -0.62988707,  0.13369188])
bl=np.array([1.84270041,  0.28992585, -0.62694901])
br=np.array([2.35213666, -0.48750511, -0.79589406])

def distance(p1,p2):
    return np.sqrt((p1-p2).dot(p1-p2))

def invec(p1,p2):
    v=p2-p1
    v=v/distance(p1,p2)
    return v
p1=tl+0.116*invec(tl,tr)
p2=tr+0.125*invec(tr,tl)
p3=bl+0.116*invec(bl,br)
p4=br+0.125*invec(br,bl)
print(distance(p1,p2))
print(distance(tl,tr))

v1=invec(tl,tr)
v2=invec(bl,br)

v=invec(p1,p3)

start=p1
end=p1+0.7*v1
start+=0.02*v
end+=0.02*v
# for i in range(10):

#     x1=np.linspace(start[0],end[0],100)
#     y1=np.linspace(start[1],end[1],100)
#     z1=np.linspace(start[2],end[2],100)
#     start+=0.1*v
#     end+=0.1*v
#     for i in range(len(x1)):
#         print("%f %f %f 255 0 0"%(x1[i],y1[i],z1[i]))

s=start+0.8*v
point3d=list()
for i in range(6):
    s+=0.1*v1
    s1=0
    s1+=s
    for i in range(8):
        point3d.append（s1)
        print("%f %f %f 0 0 255"%(s1[0],s1[1],s1[2]))
        s1+=-0.1*v
