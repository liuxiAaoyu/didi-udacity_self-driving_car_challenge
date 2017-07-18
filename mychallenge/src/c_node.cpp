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

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "point_types.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include "mychallenge/BoundingBox.h"
#include "mychallenge/MyPoseArray.h"
#include "obsstatus.hpp"
#include <std_msgs/String.h>
#include "mychallenge/MyOption.h"
#include "radar_driver/RadarTracks.h"
#include "radar_driver/Track.h"

#include<iostream>
#include<fstream>

#define IMAGE_HEIGHT	600 //800
#define IMAGE_WIDTH	600  //700
#define HEIGHT_MAX 1.27
#define HEIGHT_MIN -2.73
#define BIN		0.1

using namespace cv;


// Global Publishers/Subscribers
ros::Subscriber subBoundingBoxLidar;
ros::Subscriber subLidarPointCloud;
ros::Subscriber subRadarPointCloud;
ros::Subscriber subRadarTracks;
ros::Subscriber subBoundingBoxRadar;
ros::Publisher pubFeatureMap;
ros::Publisher pubMyPoseArray;

pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::Ptr cloud(new pcl::PointCloud<velodyne_pointcloud::PointXYZIR>);
pcl::PointCloud<pcl::PointXYZ>::Ptr Radarcloud(new pcl::PointCloud<pcl::PointXYZ>);


double heightArray[IMAGE_HEIGHT][IMAGE_WIDTH];
double lowArray[IMAGE_HEIGHT][IMAGE_WIDTH];
double tempdouble[IMAGE_HEIGHT][IMAGE_WIDTH];
double tempdouble2[IMAGE_HEIGHT][IMAGE_WIDTH];
typedef unsigned char uint8;

int LidarFrameCounter;
double lowest;

ObsStatus* myObses = new ObsStatus();

// map meters to 0->255
int map_m2i(double val) {
	if (val < HEIGHT_MIN)
		return 0;
	if (val > HEIGHT_MAX)
		return 255;
	return (int)round((val + 2.73) / (4) * 255);
}

// map meters to index
// returns 0 if not in range, 1 if in range and row/column are set
int map_pc2rc(double x, double y, int *row, int *column) {
	// Find x -> row mapping
	*row = (int)round(floor(((((IMAGE_HEIGHT*BIN) / 2.0) - x) / (IMAGE_HEIGHT*BIN)) * IMAGE_HEIGHT));	// obviously can be simplified, but leaving for debug
	// Find y -> column mapping
	*column = (int)round(floor(((((IMAGE_WIDTH*BIN) / 2.0) - y) / (IMAGE_WIDTH*BIN)) * IMAGE_WIDTH));
	// Return success
	return 1;
}

// map index to meters
// returns 0 if not in range, 1 if in range and x/y are set
int map_rc2pc(double *x, double *y, int row, int column) {
	// Check if falls within range
	if (row >= 0 && row < IMAGE_HEIGHT && column >= 0 && column < IMAGE_WIDTH) {
		// Find row -> x mapping
		*x = (double)(BIN*-1.0 * (row - (IMAGE_HEIGHT / 2.0)));	// this one is simplified
		// column -> y mapping
		*y = (double)(BIN*-1.0 * (column - (IMAGE_WIDTH / 2.0)));
		// Return success
		return 1;
	}
	return 0;
}

//calculate z value
std::pair<double, double> cal_z(double xmin, double ymin, double xmax, double ymax) {
	int xstart = xmin*IMAGE_WIDTH;
	int xend = xmax*IMAGE_WIDTH;
	int ystart = ymin*IMAGE_HEIGHT;
	int yend = ymax*IMAGE_HEIGHT;
	double zmax = -FLT_MAX;
	double zmin = FLT_MAX;
	double z;
	int count1 = 0;
	double zsmax = 0;
	double zsmin = 0;
	int conut2 = 0;
	for (int i = xstart; i <= xend; i++) {
		for (int j = ystart; j < yend; j++) {
			if (heightArray[j][i]>HEIGHT_MAX)
				continue;
			// if (heightArray[j][i] - lowArray[j][i] < 1.2){
			// 	z += heightArray[j][i]
			// 	count1++;
			// 	continue;
			// }
			if (zmax < heightArray[j][i])
				zmax = heightArray[j][i];
			// if (zmin > lowArray[j][i])
			// 	zmin = lowArray[j][i];
		}
	}
	for (int i = xstart; i <= xend; i++) {
		for (int j = ystart; j < yend; j++) {
			if (lowArray[j][i]<zmax-1.8)
				continue;
			if (zmin > lowArray[j][i])
				zmin = lowArray[j][i];
		}
	}
	if (zmax == -FLT_MAX)
		zmax = 0;
	if (zmin == FLT_MAX)
		zmin = -1.8;

	if (zmax - zmin > 1.2)
		z = (zmax + zmin) / 2;
	else
		z = 99;

	return std::make_pair(zmax, zmin);
}

double secs;
void GetLidarPointCLoud(const sensor_msgs::PointCloud2ConstPtr& pointCloudMsg)
{
  ROS_DEBUG("Point Cloud Received");
  secs =ros::Time::now().toSec();
  mychallenge::MyOption opmsg;
  opmsg.header.stamp = pointCloudMsg->header.stamp;//ros::Time::now();
  opmsg.header.frame_id = "featureMap";
  // clear cloud and height map array
  lowest = FLT_MAX;

	std::cout<<std::endl;
	std::cout<<pointCloudMsg->header.stamp.toNSec()<<"s";
  memcpy(heightArray, tempdouble, sizeof(double)*IMAGE_HEIGHT*IMAGE_WIDTH);
  memcpy(lowArray, tempdouble2, sizeof(double)*IMAGE_HEIGHT*IMAGE_WIDTH);
  
  // Convert from ROS message to PCL point cloud
  pcl::fromROSMsg(*pointCloudMsg, *cloud);


  // Populate the DEM grid by looping through every point
  int row, column;
  for(size_t j = 0; j < cloud->points.size(); ++j){
    
    // If the point is within the image size bounds
    if(map_pc2rc(cloud->points[j].x, cloud->points[j].y, &row, &column) == 1 && row >= 0 && row < IMAGE_HEIGHT && column >=0 && column < IMAGE_WIDTH){
      if(-2.5<cloud->points[j].x&& cloud->points[j].x<2.5 && -1<cloud->points[j].y&&cloud->points[j].y<1)
        continue;
      opmsg.density[row*IMAGE_WIDTH+column] += 1;
	  if (cloud->points[j].z < lowArray[row][column]) {
			lowArray[row][column] = cloud->points[j].z;
		}

      if(cloud->points[j].z > heightArray[row][column]){
        heightArray[row][column] = cloud->points[j].z;
        opmsg.height[row*IMAGE_WIDTH+column] = map_m2i(heightArray[row][column]);
        opmsg.ring[row*IMAGE_WIDTH+column] = cloud->points[j].ring;
      }
      if((uint8)cloud->points[j].intensity>opmsg.intensity[row*IMAGE_WIDTH+column])
        opmsg.intensity[row*IMAGE_WIDTH+column] = (uint8)cloud->points[j].intensity;
      // Keep track of lowest point in cloud for flood fill

    }
  }

  pubFeatureMap.publish(opmsg);
	std::cout<<"e";
}


void GetLidarBBox(mychallenge::BoundingBox Msg) {
	// if(Msg.num>0){
	//   std::cout<<Msg.header.stamp<<std::endl;
	//   std::cout<<Msg.classes[0]<<" "<<Msg.scores[0]<<std::endl;
	//   std::cout<<Msg.xmin[0]<<" "<<Msg.ymin[0]<<" "<<Msg.xmax[0]<<" "<<Msg.ymax[0]<<std::endl;
	//   std::pair<double,double> zs=cal_z(Msg.xmin[0],Msg.ymin[0],Msg.xmax[0],Msg.ymax[0]);
	//   double x=-((Msg.xmax[0]+Msg.xmin[0])/2-0.5)*IMAGE_WIDTH*BIN;
	//   double y=-((Msg.ymax[0]+Msg.ymin[0])/2-0.5)*IMAGE_HEIGHT*BIN;
	//   double z;
	//   if(zs.first-zs.second>2)
	//     z=zs.first-0.762;
	//   else
	//     z=(zs.first+zs.second)/2;
	//   std::cout<<y<<" "<<x<<" "<<std::endl;
	//   Msg.header.stamp.to_nsec();
	//   out<<Msg.header.stamp.toNSec()<<" "<<y<<" "<<x<<" "<<z<<std::endl;
	// }

	double secs1 =ros::Time::now().toSec();
	std::vector<BBox> bboxes;
	std::vector<float> vx;
	std::vector<float> vy;
	std::vector<float> vzmax;
	std::vector<float> vzmin;
	std::vector<float> vo;
	for (int i = 0; i < Msg.num; i++) {
		std::pair<double, double> zs = cal_z(Msg.xmin[i], Msg.ymin[i], Msg.xmax[i], Msg.ymax[i]);

		double x = -((Msg.xmax[i] + Msg.xmin[i]) / 2 - 0.5)*IMAGE_WIDTH*BIN;
		double y = -((Msg.ymax[i] + Msg.ymin[i]) / 2 - 0.5)*IMAGE_HEIGHT*BIN;
		double z;
		double h;
		// if (zs.first - zs.second > 2)
		// 	z = zs.first - 0.762;
		// else
		// 	z = (zs.first + zs.second) / 2;
		float orientation = Msg.orientations[i];
		bboxes.push_back(BBox(Msg.classes[i], Msg.scores[i], Msg.xmin[i], Msg.xmax[i], Msg.ymin[i], Msg.ymax[i]));
		vx.push_back(y);
		vy.push_back(x);
		vzmax.push_back(zs.first);
		vzmin.push_back(zs.second);
		vo.push_back(orientation);
	}
	LidarBBox lb = LidarBBox(Msg.header.stamp.toNSec(), Msg.num, bboxes, vx, vy, vzmax, vzmin, vo);

	myObses->addLidarBBox(lb);

	double secs2 =ros::Time::now().toSec();
	mychallenge::MyPoseArray PoseArray;
	PoseArray.header.stamp = Msg.header.stamp;
	myObses->Output(Msg.header.stamp.toNSec(),PoseArray);
	pubMyPoseArray.publish(PoseArray);
	std::cout<<"Time:  "<<secs2-secs<<std::endl;
	
}


void GetRadarPointCloud(const sensor_msgs::PointCloud2ConstPtr& pointCloudMsg)
{
	ROS_DEBUG("Point Cloud Received");
	// clear cloud and height map array
	std::vector<cv::Point3f> objectPoints;
	std::cout<<'R';

	// Convert from ROS message to PCL point cloud
	pcl::fromROSMsg(*pointCloudMsg, *Radarcloud);

	// Populate the DEM grid by looping through every point
	int row, column;
	for (size_t j = 0; j < Radarcloud->points.size(); ++j) {
		// If the point in the image size bounds
		//change the radar coordinate to velodyne coordinate
		objectPoints.push_back(cv::Point3f(Radarcloud->points[j].x + 3.8 - 1.5495, Radarcloud->points[j].y, Radarcloud->points[j].z - 1.27));
		std::cout<<Radarcloud->points[j].x<<" "<<Radarcloud->points[j].y<<" "<<Radarcloud->points[j].z-1.27<<std::endl;

	}

	std::cout<<"+++++++++++++++++++++++++++++++++++"<<std::endl;
	//myObses->addRadarPoints(RadarPoints(pointCloudMsg->header.stamp.toNSec(), objectPoints));

}

void GetRadarTracks(const radar_driver::RadarTracks Msg)
{
	long t = Msg.header.stamp.toNSec();
	std::vector<radar_driver::Track> v = Msg.tracks;
	std::vector<cv::Point3f> objectPoints;
	std::vector<float> vrange;
	std::vector<float> vrate;
	std::vector<float> vangle;
	for(int i=0; i<v.size(); i++){
		// int status = v[i].status;
		// int number = v[i].number;
		// // range is the distance from radar
		float range = v[i].range;
		if(range > 80)
			continue;
		// // Velocity of the obstacle relative to radar
		float rate = v[i].rate;
		// // Accelearation of obstacle
		// float accel = v[i].accel;
		// // Angle in radian
		float angle = v[i].angle * -1 / 180 * M_PI;
		// float width = v[i].width;
		// // Lateral velocity relative to radar
		// float late_rate = v[i].late_rate;
		// bool moving = v[i].moving;
		// float power = v[i].power;
		// float absolute_rate = v[i].absolute_rate;
		// std::cout<<" status:"<<status<<" "<<" number:"<<number<<" range:"<<range<<" rate:"<<rate<<" accel:"<<accel<<" angle:"<<angle
		// <<" width:"<<width<<" late_rate:"<<late_rate<<" moving:"<<moving<<" power:"<<power<<" absolute_rate:"<<absolute_rate<<std::endl;
		float x = (range) * cos(angle);
		float y = (range) * sin(angle);
		if( x < 28)// && y < 30 && y > -30)
			continue;
		float z = -1;
		objectPoints.push_back(cv::Point3f(x, y, z));
		vrange.push_back(range);
		vrate.push_back(rate);
		vangle.push_back(angle);
	}
	//std::cout<<"------------------------------"<<std::endl;
	myObses->addRadarPoints(RadarPoints(t, objectPoints, vrange, vangle, vrate));
	//myObses->Output(Msg.header.stamp.toNSec());
	
}

void GetImgBBox(mychallenge::BoundingBox Msg) {

	// if(Msg.num>0){
	//   std::cout<<Msg.header.stamp<<std::endl;
	//   std::cout<<Msg.classes[0]<<" "<<Msg.scores[0]<<std::endl;
	//   std::cout<<Msg.xmin[0]<<" "<<Msg.ymin[0]<<" "<<Msg.xmax[0]<<" "<<Msg.ymax[0]<<std::endl;
	// }
	std::vector<BBox> bboxes;
	for (int i = 0; i < Msg.num; i++) {
		bboxes.push_back(BBox(Msg.classes[i], Msg.scores[i], Msg.xmin[i], Msg.xmax[i], Msg.ymin[i], Msg.ymax[i]));
	}
	ImgBBox ib = ImgBBox(Msg.header.stamp.toNSec(), Msg.num, bboxes);

	myObses->addImgBBox(ImgBBox(Msg.header.stamp.toNSec(), Msg.num, bboxes));

}

void Getoption(const std_msgs::String::ConstPtr& msg) {
	ROS_INFO("Option: [%s]", msg->data.c_str());
	//myObses->Output();
}


int main(int argc, char** argv)
{
	ROS_INFO("Starting Process Node");
	ros::init(argc, argv, "c_node");
	ros::NodeHandle nh;


    // Setup image
    for(int i = 0; i < IMAGE_HEIGHT; ++i){
       for(int j = 0; j < IMAGE_WIDTH; ++j){
         tempdouble[i][j] = (double)(-FLT_MAX);
         tempdouble2[i][j] = (double)(FLT_MAX);
       }
     }


	// Setup Image Output Parameters
	LidarFrameCounter = 0;
	lowest = FLT_MAX;


	subLidarPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 2, GetLidarPointCLoud);
	subBoundingBoxLidar = nh.subscribe<mychallenge::BoundingBox>("/BoundingBox/Lidar", 2, GetLidarBBox);

	//subRadarPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/radar/points", 2, GetRadarPointCloud);
	subRadarTracks = nh.subscribe<radar_driver::RadarTracks>("/radar/tracks", 1, GetRadarTracks);
	subBoundingBoxRadar = nh.subscribe<mychallenge::BoundingBox>("/BoundingBox/Radar", 2, GetImgBBox);

	pubFeatureMap = nh.advertise<mychallenge::MyOption>("/pointcloud/featureMap", 1);
	pubMyPoseArray = nh.advertise<mychallenge::MyPoseArray>("/ObsPoseArray", 1);

	ros::Subscriber subOption = nh.subscribe("/MyOption", 2, Getoption);
	ros::spin();

	return 0;
}
