# Readme

## Solution

In the round 2 challenge, I using the neural network to process the lidar information to
get the obstacle position and pose information. First, the lidar information is projected into
a 2d bird's-eye view. Second, use the method similar to SSD to get the target category,
bounding box and orientation, and calculate the size of the target (length, width, height)
according to the bounding box size and orientation. Then, the detected target is tracked by
a Kalman filter.

Look Round2Document.pdf for more details.

I'm very sorry that I don't have enough time to define friendly interface. So, you have to change the hard code again and again.

## Usage
Firstly, we should run the c_node, change line 154 inobsstatus.hpp to you txt file loaction,which will start the main pipeline:
`$rosrun mychallenge c_node`

Then, run the python02 to process lidar feature map:
`$rosun mychallenge pythonnode2.py`

Then, replay bag:
`$rosbag play ford01.bag`

Then, generate tracklet file:
`$txt_to_tracklet.py -if yourtxtpath -o youroutdir`

Then, restart c_node (I explianed why restart in Round2Document) and do this pipeline again for other bag.


