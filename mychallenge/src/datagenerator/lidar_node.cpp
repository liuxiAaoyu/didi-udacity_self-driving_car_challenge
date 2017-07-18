// ROS Point Cloud DEM Generation
// MacCallister Higgins

#include <cmath>
#include <vector>
#include <float.h>
#include <stdio.h>
#include <math.h> 
#include <sstream>
#include <string.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "/home/xiaoyu/catkin_ws/src/velodyne/velodyne_pointcloud/include/velodyne_pointcloud/point_types.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>

#include<iostream>
#include<fstream>

#define IMAGE_HEIGHT	600  //800
#define IMAGE_WIDTH	600  //700
#define HEIGHT_MAX 1.27
#define HEIGHT_MIN -2.73
#define BIN		0.1

using namespace cv;
typedef unsigned char uint8;
// Global Publishers/Subscribers
ros::Subscriber subPointCloud;
ros::Publisher pubHeightMap;

pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::Ptr cloud (new pcl::PointCloud<velodyne_pointcloud::PointXYZIR>);
sensor_msgs::PointCloud2 output;

double heightArray[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 heightFeatureArray[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 ringArray[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 densityArray[IMAGE_HEIGHT][IMAGE_WIDTH];

uint8 slicesyArray0[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 slicesyArray1[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 slicesyArray2[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 slicesyArray3[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 slicesyArray4[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 slicesyArray5[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 slicesyArray6[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 slicesyArray7[IMAGE_HEIGHT][IMAGE_WIDTH];


double tempdouble[IMAGE_HEIGHT][IMAGE_WIDTH];
uint8 tempint[IMAGE_HEIGHT][IMAGE_WIDTH];


cv::Mat *heightmap;
vector<int> compression_params;

int fnameCounter;
double lowest;
double average;

// map meters to 0->255
int map_m2i(double val){
  if (val<HEIGHT_MIN)
    return 0;
  if (val>HEIGHT_MAX)
    return 255;
  return (int)round((val + 2.73 )/(4) * 255);
}

// map meters to index
// returns 0 if not in range, 1 if in range and row/column are set
int map_pc2rc(double x, double y, int *row, int *column){
    // Find x -> row mapping
    *row = (int)round(floor(((((IMAGE_HEIGHT*BIN)/2.0) - x)/(IMAGE_HEIGHT*BIN)) * IMAGE_HEIGHT));	// obviously can be simplified, but leaving for debug

    // Find y -> column mapping
    *column = (int)round(floor(((((IMAGE_WIDTH*BIN)/2.0) - y)/(IMAGE_WIDTH*BIN)) * IMAGE_WIDTH));

    // Return success
    return 1;
}

// map index to meters
// returns 0 if not in range, 1 if in range and x/y are set
int map_rc2pc(double *x, double *y, int row, int column){
  // Check if falls within range
  if(row >= 0 && row < IMAGE_HEIGHT && column >= 0 && column < IMAGE_WIDTH){
    // Find row -> x mapping
    *x = (double)(BIN*-1.0 * (row - (IMAGE_HEIGHT/2.0)));	// this one is simplified

    // column -> y mapping
    *y = (double)(BIN*-1.0 * (column - (IMAGE_WIDTH/2.0)));

    // Return success
    return 1;
  }

  return 0;
}

// main generation function
void feature11(const sensor_msgs::PointCloud2ConstPtr& pointCloudMsg)
{
  ROS_DEBUG("Point Cloud Received");
  double secs =ros::Time::now().toSec();
  // clear cloud and height map array
  lowest = FLT_MAX;
  // for(int i = 0; i < IMAGE_HEIGHT; ++i){ 
  //   for(int j = 0; j < IMAGE_WIDTH; ++j){ 
  //     heightArray[i][j] = (double)(-FLT_MAX);
  //     ringArray[i][j] = 0;
  //     densityArray[i][j] = 0;
  //     slicesyArray0[i][j] = 0;
  //     slicesyArray1[i][j] = 0;
  //     slicesyArray2[i][j] = 0;
  //     slicesyArray3[i][j] = 0;
  //     slicesyArray4[i][j] = 0;
  //     slicesyArray5[i][j] = 0;
  //     slicesyArray6[i][j] = 0;
  //     slicesyArray7[i][j] = 0;
  //     }
  //   }
  memcpy(heightArray, tempdouble, sizeof(double)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(ringArray, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(densityArray, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(slicesyArray0, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(slicesyArray1, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(slicesyArray2, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(slicesyArray3, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(slicesyArray4, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(slicesyArray5, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(slicesyArray6, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(slicesyArray7, tempint, sizeof(uint8)*IMAGE_HEIGHT*IMAGE_WIDTH);
  // Convert from ROS message to PCL point cloud
  pcl::fromROSMsg(*pointCloudMsg, *cloud);


  // Populate the DEM grid by looping through every point
  int row, column;
  for(size_t j = 0; j < cloud->points.size(); ++j){
    
    // If the point is within the image size bounds
    if(map_pc2rc(cloud->points[j].x, cloud->points[j].y, &row, &column) == 1 && row >= 0 && row < IMAGE_HEIGHT && column >=0 && column < IMAGE_WIDTH){
      if(-2.5<cloud->points[j].x&& cloud->points[j].x<2.5 && -1<cloud->points[j].y&&cloud->points[j].y<1)
        continue;
      densityArray[row][column] += 1;
      double tempheight = cloud->points[j].z;
      int tempintensity = (int)cloud->points[j].intensity;
      if(-2.73<tempheight<-2.23)
        if(tempintensity>slicesyArray0[row][column])
          slicesyArray0[row][column] = tempintensity;
      if(-2.23<tempheight<-1.73)
        if(tempintensity>slicesyArray0[row][column])
          slicesyArray1[row][column] = tempintensity;
      if(-1.73<tempheight<-1.23)
        if(tempintensity>slicesyArray0[row][column])
          slicesyArray2[row][column] = tempintensity; 
      if(-1.23<tempheight<-0.73)
        if(tempintensity>slicesyArray0[row][column])
          slicesyArray3[row][column] = tempintensity; 
      if(-0.73<tempheight<-0.23)
        if(tempintensity>slicesyArray0[row][column])
          slicesyArray4[row][column] = tempintensity; 
      if(-0.23<tempheight<0.27)
        if(tempintensity>slicesyArray0[row][column])
          slicesyArray5[row][column] = tempintensity; 
      if(0.27<tempheight<0.77)
        if(tempintensity>slicesyArray0[row][column])
          slicesyArray6[row][column] = tempintensity; 
      if(0.77<tempheight<1.27)
        if(tempintensity>slicesyArray0[row][column])
          slicesyArray7[row][column] = tempintensity; 


      if(cloud->points[j].z > heightArray[row][column]){
        heightArray[row][column] = cloud->points[j].z;
        heightFeatureArray[row][column] = map_m2i(heightArray[row][column]);
        ringArray[row][column] = cloud->points[j].ring;
      }
      // Keep track of lowest point in cloud for flood fill

    }
  }


  // for(int i = 0; i < IMAGE_HEIGHT; ++i){ 
  //   for(int j = 0; j < IMAGE_WIDTH; ++j){ 
  //     heightFeatureArray[i][j] = map_m2i(heightArray[i][j]);
  //   }
  // }
  // for(int i = 0; i < IMAGE_HEIGHT; ++i){ 
  //   for(int j = 0; j < IMAGE_WIDTH; ++j){ 

  //     // Add point to image
  //     cv::Vec3b &pixel = heightmap->at<cv::Vec3b>(i,j);
  //     if(heightArray[i][j] > -FLT_MAX){
  //       pixel[2] = densityArray[i][j]>255?255:densityArray[i][j];
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
  ++fnameCounter;

  
  sensor_msgs::Image im;
  cv_bridge::CvImage cvi;
  cvi.header.stamp=pointCloudMsg->header.stamp;//ros::Time::now();
  cvi.header.frame_id="heightMap";
  cvi.encoding="bgr8";
  cvi.image=*heightmap;
  cvi.toImageMsg(im);
  pubHeightMap.publish(im);
  double secs1 =ros::Time::now().toSec();
  average+=secs1-secs;
  double aa=average/fnameCounter;
  std::cout<<secs1-secs<<" "<<aa<<std::endl;
  std::cout<<ros::Time::now().toNSec()<<std::endl;
}

// void DEM2(const sensor_msgs::PointCloud2ConstPtr& pointCloudMsg)
// {
//   ROS_DEBUG("Point Cloud Received");
//   double secs =ros::Time::now().toSec();
//   // clear cloud and height map array
//   lowest = FLT_MAX;
//   for(int i = 0; i < IMAGE_HEIGHT; ++i){ 
//     for(int j = 0; j < IMAGE_WIDTH; ++j){ 
//       heightArray[i][j] = (double)(-FLT_MAX);
//       ringArray[i][j] = 0;
//       densityArray[i][j] = 0;
//       }
//     }

//   pcl::fromROSMsg(*pointCloudMsg, *cloud);


//   // Populate the DEM grid by looping through every point
//   int row, column;
//   for(size_t j = 0; j < cloud->points.size(); ++j){
    
//     // If the point is within the image size bounds
//     if(map_pc2rc(cloud->points[j].x, cloud->points[j].y, &row, &column) == 1 && row >= 0 && row < IMAGE_HEIGHT && column >=0 && column < IMAGE_WIDTH){
//       if(-2.5<cloud->points[j].x&& cloud->points[j].x<2.5 && -1<cloud->points[j].y&&cloud->points[j].y<1)
//         continue;
//         densityArray[row][column] += 1;
//       if(cloud->points[j].z > heightArray[row][column]){
//         heightArray[row][column] = cloud->points[j].z;
//         ringArray[row][column] = cloud->points[j].ring;
//         }
//       // Keep track of lowest point in cloud for flood fill
//       else if(cloud->points[j].z < lowest){
//         lowest = cloud->points[j].z;
//         }
//       }
//     }


//   for(int i = 0; i < IMAGE_HEIGHT; ++i){ 
//     for(int j = 0; j < IMAGE_WIDTH; ++j){ 

//       // Add point to image
//       cv::Vec3b &pixel = heightmap->at<cv::Vec3b>(i,j);
//       if(heightArray[i][j] > -FLT_MAX){
//         pixel[2] = densityArray[i][j]>255?255:densityArray[i][j];
//         pixel[1] = ringArray[i][j] ;
//         pixel[0] = map_m2i(heightArray[i][j]);
//         }
//       else{
//         pixel[0] = 0;
//         pixel[1] = 0;
//         pixel[2] = 0;//map_m2i(lowest);
//         }
//       }
//     }
//   // Display iamge
//   //cv::imshow("Height Map", *heightmap);

//   // Save image to disk
//   //char filename[100];
//   //snprintf(filename, 100, "/home/xiaoyu/Documents/1DIDIUDA/ros-examples/images/image_%d.png", fnameCounter);
//   //cv::imwrite(filename, *heightmap, compression_params);
//   ++fnameCounter;

  
//   sensor_msgs::Image im;
//   cv_bridge::CvImage cvi;
//   cvi.header.stamp=pointCloudMsg->header.stamp;//ros::Time::now();
//   cvi.header.frame_id="heightMap";
//   cvi.encoding="bgr8";
//   cvi.image=*heightmap;
//   cvi.toImageMsg(im);
//   pubHeightMap.publish(im);
//   double secs1 =ros::Time::now().toSec();
//   average+=secs1-secs;
//   double aa=average/fnameCounter;
//   std::cout<<secs1-secs<<" "<<aa<<std::endl;
// }

int main(int argc, char** argv)
{
  ROS_INFO("Starting LIDAR Node");
  ros::init(argc, argv, "lidar_node");
  ros::NodeHandle nh;

  for(int i = 0; i < IMAGE_HEIGHT; ++i){ 
    for(int j = 0; j < IMAGE_WIDTH; ++j){ 
      tempdouble[i][j] = (double)(-FLT_MAX);
      tempint[i][j] = 0;
    }
  }

  // Setup image
  cv::Mat map(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
  heightmap = &map;
  //cvNamedWindow("Height Map", CV_WINDOW_AUTOSIZE);
  //cvStartWindowThread();
  //cv::imshow("Height Map", *heightmap);

  // Setup Image Output Parameters
  fnameCounter = 0;
  average = 0;
  lowest = FLT_MAX;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
 
  // Setup indicies in point clouds


  subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 2, feature11);
  //pubHeightMap = nh.advertise<sensor_msgs::PointCloud2> ("/heightmap/pointcloud", 1);
  pubHeightMap = nh.advertise<sensor_msgs::Image>("/heightmap/pointcloud",1);
  ros::spin();

  return 0;
}
