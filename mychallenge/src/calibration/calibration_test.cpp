// # Copyright 2017 LIU XIAOYU. All Rights Reserved.
// #
// # Licensed under the Apache License, Version 2.0 (the "License");
// # you may not use this file except in compliance with the License.
// # You may obtain a copy of the License at
// #
// # http://www.apache.org/licenses/LICENSE-2.0
// #
// # Unless required by applicable law or agreed to in writing, software
// # distributed under the License is distributed on an "AS IS" BASIS,
// # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// # See the License for the specific language governing permissions and
// # limitations under the License.
// # ==============================================================================

#include <cmath>
#include <vector>
#include <float.h>
#include <stdio.h>
#include <math.h> 
#include <sstream>
#include <string.h>

#include <image_transport/image_transport.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "/home/xiaoyu/catkin_ws/src/velodyne/velodyne_pointcloud/include/velodyne_pointcloud/point_types.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>

#include <fstream>

#include <iostream>
//#include <pcl/console/parse.h>
//#include <pcl/filters/extract_indices.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <pcl/sample_consensus/ransac.h>
//#include <pcl/sample_consensus/sac_model_plane.h>
//#include <pcl/sample_consensus/sac_model_sphere.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <boost/thread/thread.hpp>

//#include <pcl/ModelCoefficients.h>
//#include <pcl/segmentation/sac_segmentation.h>


#define IMAGE_HEIGHT	800
#define IMAGE_WIDTH	700
#define HEIGHT_MAX 1.27
#define HEIGHT_MIN -2.73
#define BIN		0.1

using namespace cv;

// Global Publishers/Subscribers
ros::Subscriber subPointCloud;

pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::Ptr cloud (new pcl::PointCloud<velodyne_pointcloud::PointXYZIR>);

cv::Mat *heightmap;
//vector<int> compression_params;

int fnameCounter;
double lowest;
std::vector<cv::Point2f> imagePoints;

double rx,ry,rz;
//boost::shared_ptr<pcl::visualization::PCLVisualizer>
//simpleVis (pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud)
//{
//  // --------------------------------------------
//  // -----Open 3D viewer and add point cloud-----
//  // --------------------------------------------
//  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
//  viewer->setBackgroundColor (0, 0, 0);
//  viewer->addPointCloud<pcl::PointXYZI> (cloud, "sample cloud");
//  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
//  //viewer->addCoordinateSystem (1.0, "global");
//  viewer->initCameraParameters ();
//  return (viewer);
//}


// main generation function
void test(const sensor_msgs::PointCloud2ConstPtr& pointCloudMsg)
{
  ROS_INFO("Point Cloud Received");

  std::vector<cv::Point3f> objectPoints;
  // Convert from ROS message to PCL point cloud
  pcl::fromROSMsg(*pointCloudMsg, *cloud);


  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_new (new pcl::PointCloud<pcl::PointXYZI>);
  // Populate the DEM grid by looping through every point
  int row, column;
  int index=0;
  for(size_t j = 0; j < cloud->points.size(); ++j){
    
      if(1<cloud->points[j].x){
        objectPoints.push_back(cv::Point3f(cloud->points[j].x,cloud->points[j].y,cloud->points[j].z));
        index++;
        continue;
      }
       // objectPoints.push_back(cv::Point3f(cloud->points[j].x,cloud->points[j].y,cloud->points[j].z));
  }

  // //height map
  // for(int i = 0; i < IMAGE_HEIGHT; ++i){ 
  //   for(int j = 0; j < IMAGE_WIDTH; ++j){ 
  //     // Add point to image
  //     cv::Vec3b &pixel = heightmap->at<cv::Vec3b>(i,j);
  //     if(heightArray[i][j] > -FLT_MAX){
  //       pixel[2] = densityArray[i][j];
  //       pixel[1] = ringArray[i][j] ;
  //       pixel[0] = map_m2i(heightArray[i][j]);
  //       }
  //     else{
  //       pixel[0] = 0;
  //       pixel[1] = 0;
  //       pixel[2] = 0;//map_m2i(lowest);
  //       }
  //     }
  //   }
  // Display iamge
  //cv::imshow("Height Map", *heightmap);

  // Save image to disk
  //char filename[100];
  //snprintf(filename, 100, "/home/xiaoyu/Documents/1DIDIUDA/ros-examples/images/image_%d.png", fnameCounter);
  //cv::imwrite(filename, *heightmap, compression_params);
//   std::string fname;
//   std::stringstream ss;
//   ss << pointCloudMsg->header.stamp;
//   ss >> fname;
//   //if(fnameCounter==1){
//     std::string str = "/home/xiaoyu/Documents/"+fname+".txt";
//     std::ofstream fout(str.c_str());
//     for(size_t j = 0; j < cloud->points.size(); ++j){
//       fout<<cloud->points[j].x<<" "<<cloud->points[j].y<<" "<<cloud->points[j].z<<
//         " "<<cloud->points[j].intensity<<" "<<cloud->points[j].intensity<<" "<<cloud->points[j].intensity<<std::endl;
//     }
//     fout.close();
//     //exit(1);
//   //}


 
//   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;  
//   viewer = simpleVis(cloud_final);
//   while (!viewer->wasStopped ())
//   {
//     viewer->spinOnce (100);
//     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//   }
  ++fnameCounter;
  std::cout<<fnameCounter<<std::endl;

  cv::Mat m_cam = (Mat_<double>(3,3) << 1394.62, 0.000000, 625.888005,0.000000, 1568.65, 559.626310,0,0,1);// 1384.621562, 0.000000, 625.888005,0.000000, 1393.652271, 559.626310,0,0,1);
  cv::Mat m_dist = (Mat_<double>(5,1) << 0, 0, 0, 0, 0);// -0.152089, 0.270168, 0.003143, -0.005640, 0.000000);
  cv::Mat m_rvecs = (Mat_<double>(3,1) <<1.20914, -1.22553, 1.18114 );//   1.18913633 -1.23253128 1.2014496  //1.20513633 -1.22253128 1.17414496  //1.18513633 -1.17253128 1.19414496
  cv::Mat m_tvecs = (Mat_<double>(3,1) << -0.04032959, -0.40270426, -0.47190471);

//  int key = cv::waitKey(0);
//  switch (key)  {
//    case 'q':
//      rx += 0.01;
//      break;
//  case 'w':
//    rx -= 0.01;
//    break;
//  case 'a':
//    ry += 0.01;
//    break;
//  case 's':
//    ry -= 0.01;
//    break;
//  case 'z':
//    rz += 0.01;
//    break;
//  case 'x':
//    ry -= 0.01;
//    break;
//  }

  imagePoints.clear();
  cv::projectPoints(objectPoints, m_rvecs, m_tvecs, m_cam, m_dist, imagePoints);




}

void getImage(const sensor_msgs::ImageConstPtr& msg){
    *heightmap=cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::Mat img_projected;
    heightmap->copyTo(img_projected);

//    std::string fname;
//    std::stringstream ss;
//    ss << msg->header.stamp;
//    ss >> fname;
//    std::string str = "/home/xiaoyu/Documents/"+fname+".jpg";
//    imwrite(str,img_projected);


    for (int i =0; i<imagePoints.size();i++){
      int x=(int)imagePoints[i].x;
      int y=(int)imagePoints[i].y;
      if(0<=x&&x<1368&&y>=0&&y<1096){
        img_projected.at<cv::Vec3b>(y,x)[2]=255;
        img_projected.at<cv::Vec3b>(y,x)[0]=255;
        img_projected.at<cv::Vec3b>(y,x)[1]=255;
      }
    }

    imwrite("/home/xiaoyu/round.jpg",img_projected);
    //cv::pyrDown(img_projected,img_projected);
    imshow("img",img_projected);

}

int main(int argc, char** argv)
{
  ROS_INFO("Starting calibration test Node");
  ros::init(argc, argv, "cal_test_node");
  ros::NodeHandle nh;

  // Setup image
  cv::Mat map(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
  heightmap = &map;
  cvNamedWindow("img", CV_WINDOW_AUTOSIZE);
  cvStartWindowThread();
  //cv::imshow("Height Map", *heightmap);

  // Setup Image Output Parameters
  fnameCounter = 0;
  //std::cin>>rx>>ry>>rz;

  subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, test);
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/image_raw", 1, getImage);
  ros::spin();

  return 0;
}
