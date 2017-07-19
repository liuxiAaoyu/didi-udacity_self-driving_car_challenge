#!/usr/bin/python

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic
import argparse
import time
import rospy
import sys
from std_msgs.msg import String



def talker():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--option', type=str, nargs='?', default='o',
        help='Output folder')
    parser.add_argument('-d', '--filedir', type=str, nargs='?', default='/1./19/',
        help='Output folder')
    args = parser.parse_args()
    option = args.option
    filedir = args.filedir


    pub = rospy.Publisher('/MyOption', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.2) # 10hz

    string = "first"
    rospy.loginfo(string)
    pub.publish(string)
    time.sleep(1)
    if option=="o":
        hello_str = "output %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
    if option == "f":
        hello_str = "%s" % filedir
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
    if option == "w":
        hello_str = "write"
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
    sys.exit(0)

    # while not rospy.is_shutdown():
    #     hello_str = "output %s" % rospy.get_time()
    #     rospy.loginfo(hello_str)
    #     pub.publish(hello_str)
    #     rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
