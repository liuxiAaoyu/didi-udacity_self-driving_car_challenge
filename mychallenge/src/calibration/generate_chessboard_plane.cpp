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

#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/project_inliers.h>


using namespace cv;

// Global Publishers/Subscribers
ros::Subscriber subPointCloud;
ros::Publisher pubPointCloud;
ros::Publisher pubHeightMap;

pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::Ptr cloud (new pcl::PointCloud<velodyne_pointcloud::PointXYZIR>);
pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::Ptr cloud_grid (new pcl::PointCloud<velodyne_pointcloud::PointXYZIR>);

pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_final (new pcl::PointCloud<pcl::PointXYZI>);



boost::shared_ptr<pcl::visualization::PCLVisualizer>
simpleVis (pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZI> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  //viewer->addCoordinateSystem (1.0, "global");
  viewer->initCameraParameters ();
  return (viewer);
}


// main generation function
void cal()
{
  ROS_INFO("Point Cloud Received");

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_new (new pcl::PointCloud<pcl::PointXYZI>);
  // Populate the DEM grid by looping through every point
  ifstream  fin("/home/xiaoyu/Documents/1.txt");
  int row, column;
  int index=0;
  if (! fin.is_open())  
       { std::cout << "Error opening file"; exit (1); } 
  while(!fin.eof()){

    std::string line;
    getline(fin,line);

    if( line[0]=='p')
        continue;
    if( line[0]=='#')
        break;
    std::stringstream ss;
    ss<<line.substr(2,line.size());
    float x,y,z;
    int i;
    ss>>x>>y>>z>>i;
    pcl::PointXYZI point_temp;
    point_temp.x = x;
    point_temp.y = y;
    point_temp.z = z;
    point_temp.intensity = i;
    cloud_new->push_back(point_temp);
    std::cout<<x<<" "<<y<<" "<<z<<std::endl;
    index++;
  }
  std::cout<<index<<std::endl;




  //std::vector<int>  inliers;
  pcl::SampleConsensusModelPlane<pcl::PointXYZI>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZI> (cloud_new));
  pcl::RandomSampleConsensus<pcl::PointXYZI> ransac (model_p);
//   ransac.setDistanceThreshold (0.2);
//   ransac.computeModel();
//   ransac.getInliers(inliers);

//   std::cout<<inliers.size()<<std::endl;;

//   // copies all inliers of the model computed to another PointCloud
//   pcl::copyPointCloud<pcl::PointXYZI>(*cloud_new, inliers, *cloud_final);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZI> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.03);

  seg.setInputCloud (cloud_new);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return ;
  }

  std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;
//0.1 Model coefficients: 0.8411 0.540749 -0.0118584 -1.71685
//0.05 Model coefficients: 0.839261 0.543584 -0.012515 -1.71307
//0.01 Model coefficients: 0.835845 0.548948 0.00441758 -1.69578
//0.03 Model coefficients: 0.832747 0.553489 -0.0135263 -1.69967

pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZI>);
pcl::copyPointCloud<pcl::PointXYZI>(*cloud_new, *inliers, *cloud_temp);
// Create the filtering object
  pcl::ProjectInliers<pcl::PointXYZI> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);
  proj.setInputCloud (cloud_temp);
  proj.setModelCoefficients (coefficients);
  proj.filter (*cloud_final);


  std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
 
 //pcl::copyPointCloud<pcl::PointXYZI>(*cloud_new, *inliers, *cloud_final);
 std::string str = "/home/xiaoyu/Documents/1.03.txt";
    std::ofstream fout(str.c_str());
    for(size_t j = 0; j < cloud_final->points.size(); ++j){
      fout<<cloud_final->points[j].x<<" "<<cloud_final->points[j].y<<" "<<cloud_final->points[j].z<<
        " "<<cloud_final->points[j].intensity<<" "<<cloud_final->points[j].intensity<<" "<<cloud_final->points[j].intensity<<std::endl;
    }
    fout.close();

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;  
  viewer = simpleVis(cloud_final);
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

}

int main(int argc, char** argv)
{
  ROS_INFO("Starting my Node");
  ros::init(argc, argv, "mychallenge_node");
  ros::NodeHandle nh;
  cal();
 
  ros::spin();

  return 0;
}
