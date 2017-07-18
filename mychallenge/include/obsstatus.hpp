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
#include<iostream>
#include<vector>
#include<queue>
#include<opencv2/opencv.hpp>
#include<cmath>
#include<map>
#include<fstream>
#include"obstacle.hpp"
#include "mychallenge/MyPoseArray.h"


#define IMG_SCORE_THRESHOLD 0.1
#define LIDAR_SCORE_THRESHOLD 0.1
#define IMG_HEIGHT 1096
#define IMG_WIDTH 1368
#define LIDAR_IMG_HEIGHT 600
#define LIDAR_IMG_WIDTH 600

//Bounding Box class
class BBox {
public:
	BBox() {};
	~BBox() {};
	BBox(int tc, float ts, float txi, float txa, float tyi, float tya) {
		clas = tc;
		score = ts;
		xmin = txi;
		xmax = txa;
		ymin = tyi;
		ymax = tya;
	};
	BBox(const BBox& bbox) {
		clas = bbox.clas;
		score = bbox.score;
		xmin = bbox.xmin;
		xmax = bbox.xmax;
		ymin = bbox.ymin;
		ymax = bbox.ymax;
	};

	int clas;
	float score;
	float xmin;
	float xmax;
	float ymin;
	float ymax;
};

// Image Bounding Box class
class ImgBBox {
public:
	ImgBBox() {};
	~ImgBBox() {};
	ImgBBox(long t, int n, std::vector<BBox> &tbboxes) {
		timestamp = t;
		num = n;
		bboxes = tbboxes;
	}
	long timestamp;
	int num;
	std::vector<BBox> bboxes;
};

//Lidar Bounding Box class
class LidarBBox {
public:
	LidarBBox() {};
	~LidarBBox() {};
	LidarBBox(long tt, int tnum, std::vector<BBox> tbboxes,
		std::vector<float> tx, std::vector<float> ty, std::vector<float> tzmax,
		std::vector<float> tzmin, std::vector<float> to) {
		timestamp = tt;
		num = tnum;
		bboxes = tbboxes;
		x = tx;
		y = ty;
		zmax = tzmax;
		zmin = tzmin;
		ori = to;
	}
	long timestamp;
	int num;
	std::vector<BBox> bboxes;
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> zmax;
	std::vector<float> zmin;
	std::vector<float> ori;
};

//Radar points class
class RadarPoints {
public:
	RadarPoints() {};
	RadarPoints(long tt, std::vector<cv::Point3f> tp, std::vector<float> tr, 
		std::vector<float> ta, std::vector<float> trr) {
		timestamp = tt;
		points = tp;
		range = tr;
		angle = ta;
		rate = trr;
	};
	~RadarPoints() {};
	long timestamp;
	std::vector<cv::Point3f> points;
	std::vector<float> range;
	std::vector<float> angle;
	std::vector<float> rate;

};


//obstacle status class
//use this class to store obstacles I tracking
class ObsStatus {
public:
	int m_iTracknum;
	cv::Mat m_mcam;
	cv::Mat m_mdist;
	cv::Mat m_mrvecs;
	cv::Mat m_mtvecs;
	std::queue<long > m_lTimestamp;
	std::queue<ImgBBox> m_qImgbbox;
	std::queue<LidarBBox> m_qLidarbbox;
	std::queue<RadarPoints> m_qRadpoints;

	int m_iImgFrameID;
	int m_iLidarFrameID;

	std::map<int, Obstacle*> m_mObstacle;//not use 
	std::map <int, Obstacle*>::iterator m_mIter;// not use
	std::list<Obstacle*> m_lObstacle; 
	
	std::list<Obstacle*>::iterator m_lIter;

	std::ofstream out;

	ObsStatus() {
		m_iTracknum = 0;
		m_iImgFrameID = 0;
		m_iLidarFrameID = 0;
		//m_mcam = (cv::Mat_<double>(3, 3) << 1384.621562, 0.000000, 625.888005, 0.000000, 1393.652271, 559.626310, 0, 0, 1);
		m_mcam = (cv::Mat_<double>(3, 3) << 1399.62, 0.000000, 625.888005, 0.000000, 1528.65, 559.626310, 0, 0, 1);
		m_mdist = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
		// std::ifstream in("/home/xiaoyu/catkin_ws/src/mychallenge/include/rt.txt");
		// if (!in.is_open()) {
		// 	std::cout << "Error opening calibration file" << std::endl;
		// 	exit(1);
		// }
		// double r1, r2, r3, t1, t2, t3;
		// in >> r1 >> r2 >> r3 >> t1 >> t2 >> t3;
		m_mrvecs = (cv::Mat_<double>(3, 1) << 1.20514, -1.22953, 1.17914);
		m_mtvecs = (cv::Mat_<double>(3, 1) << -0.04032959, -0.40270426, -0.47190471);
		//change this to you file
		out.open("/home/xiaoyu/lidar.txt");

		// cvNamedWindow("IMG", CV_WINDOW_AUTOSIZE);
		// cvStartWindowThread();
	};
	~ObsStatus() {};

	void updateTimeStamp(long timestamp, int i) {
		m_lTimestamp.push(timestamp);

	};

	void addRadarPoints(RadarPoints radpoints) {
		m_qRadpoints.push(radpoints);
		if (m_qRadpoints.size() > 20)
			m_qRadpoints.pop();
	};

	void addLidarBBox(LidarBBox lidbbox) {
		m_qLidarbbox.push(lidbbox);
		if (m_qLidarbbox.size() > 20)
			m_qLidarbbox.pop();
		long _ttimestamp = lidbbox.timestamp;
		std::cout << "Get LidatBBox "<<lidbbox.num<<" ";
		for (int i = 0; i < lidbbox.num; i++) {
			int xmin = (int)(lidbbox.bboxes[i].xmin*LIDAR_IMG_WIDTH);
			int xmax = (int)(lidbbox.bboxes[i].xmax*LIDAR_IMG_WIDTH);
			int ymin = (int)(lidbbox.bboxes[i].ymin*LIDAR_IMG_HEIGHT);
			int ymax = (int)(lidbbox.bboxes[i].ymax*LIDAR_IMG_HEIGHT);
			int lclass_id = lidbbox.bboxes[i].clas;
			float lx = lidbbox.x[i];
			float ly = lidbbox.y[i];
			float lzmax = lidbbox.zmax[i];
			float lzmin = lidbbox.zmin[i];
			float lz = (lzmax + lzmin)/2;
			float lo = lidbbox.ori[i];
			float tboxw = xmax - xmin;
			float tboxh = ymax - ymin;

			//ObsLocation tempObsLocation(_ttimestamp, 1, lx, ly, lz, lo);

			//find the obstacle of this lidar point
			int obsid = -1;
			float distance = FLT_MAX;
			std::list<Obstacle*>::iterator t_lIter;
			for (m_lIter = m_lObstacle.begin(); m_lIter != m_lObstacle.end(); ) {
				if (_ttimestamp - (*m_lIter)->m_pre_timestamp > 15e8) {

					delete *m_lIter;
					//m_lObstacleStor.push_back(*m_lIter);
					m_lObstacle.erase(m_lIter++);
					std::cout<<"X";
					continue;
				}
				float tempx=0,tempy=0,tempyaw=0;
				(*m_lIter)->GetLocation(_ttimestamp, tempx, tempy, tempyaw);
				float distx = lx - tempx;
				float disty = ly - tempy;
				float tempdistance = sqrt(distx*distx + disty*disty );
				if (tempdistance < 5 && tempdistance < distance) {
					distance = tempdistance;
					obsid = (*m_lIter)->m_id;
					t_lIter = m_lIter;
				}
				m_lIter++;
			}
			std::cout << "#" << obsid;
			if (obsid == -1) {
				//add the new location to obstacle map
				Obstacle *tempobs = new Obstacle();
				tempobs->Init(m_iTracknum, lclass_id, _ttimestamp,
				 	lx, ly, lzmax, lzmin, lo, tboxw, tboxh);
				m_lObstacle.push_back(tempobs);
				m_iTracknum++;

			}
			else {
				//update obstacles map
				(*t_lIter)->UpdateMeasureLidar(_ttimestamp, lx, ly, lzmax, ///////////////////////////////////////
					lzmin, lo, tboxw, tboxh);

			}

		}
		std::cout << std::endl;
	};

	void addImgBBox(ImgBBox imgbbox) {
		m_qImgbbox.push(imgbbox);
		if (m_qImgbbox.size() > 20)
			m_qImgbbox.pop();
		long _ttimestamp = imgbbox.timestamp;
		//for every bounding box, find corresponding radar points
		std::cout << "get ImgBBox ";
		for (int i = 0; i < imgbbox.num; i++) {
			int xmin = (int)(imgbbox.bboxes[i].xmin*IMG_WIDTH);
			int xmax = (int)(imgbbox.bboxes[i].xmax*IMG_WIDTH);
			int ymin = (int)(imgbbox.bboxes[i].ymin*(IMG_HEIGHT / 2) + IMG_HEIGHT / 2 -200);
			int ymax = (int)(imgbbox.bboxes[i].ymax*(IMG_HEIGHT / 2) + IMG_HEIGHT / 2 -200);
			int lclass_id = 0;
			long lt;
			float lx=0, ly=0, lz=0;
			float lr=0, lphi=0, lr_dot=0;
			if( imgbbox.bboxes[i].clas == 1){
				lclass_id = 1;
			}
			else
				continue;

			std::vector<cv::Point2f> imagePoints;
			std::vector<cv::Point3f> objectPoints;
			if (m_qRadpoints.size() == 0)
				break;
			objectPoints = m_qRadpoints.back().points;
			int RadPointNum = objectPoints.size();
			if (RadPointNum == 0)
				break;
			std::vector<float> range = m_qRadpoints.back().range;
			std::vector<float> angle = m_qRadpoints.back().angle;
			std::vector<float> rate = m_qRadpoints.back().rate;

			cv::projectPoints(objectPoints, m_mrvecs, m_mtvecs, m_mcam, m_mdist, imagePoints);
			// cv::Mat img_projected(1096,1368, CV_8UC3, cv::Scalar(0,0,0));

			//finding the closest radar point in bounding box
			float distance = FLT_MAX;
			int index = -1;
			for (int j = 0; j < RadPointNum; j++) {
				float x = (float)imagePoints[j].x;
				float y = (float)imagePoints[j].y;
				//std::cout<<"xy "<<imagePoints[j].x<<" "<<imagePoints[j].y;
				// cv::circle(img_projected,cv::Point2f(x,y),10,cv::Scalar(255,255,0),2);
				if (xmin-20 < x&&x < xmax+20&&ymin-10 < y&&y < ymax+10) {
					float temp = sqrt(range[j]);
					if (distance > temp) {
						distance = temp;
						index = j;
					}
				}
			}
			std::cout << "#";
			// cv::rectangle(img_projected,cv::Point2f(xmin,ymin),cv::Point2f(xmax,ymax),cv::Scalar(255,255,255),2,8);
			// cv::pyrDown(img_projected,img_projected);
			// cv::imshow("IMG",img_projected);
			//std::cout<<"index "<<index<<std::endl;
			if (index == -1)
				continue;
			std::cout << "#";
			//std::cout<<objectPoints[index].x<<"    "<<objectPoints[index].y<<"    "<<objectPoints[index].z<<" "<<std::endl;
			lt = m_qRadpoints.back().timestamp;
			lx = objectPoints[index].x;
			ly = objectPoints[index].y;
			lz = objectPoints[index].z;
			lr = range[index];
			lphi = angle[index];
			lr_dot = rate[index];

			//find the obstacle of this radar point
			int obsid = -1;
			distance = FLT_MAX;
			std::list<Obstacle*>::iterator t_lIter;
			for (m_lIter = m_lObstacle.begin(); m_lIter != m_lObstacle.end(); ) {
				if (_ttimestamp - (*m_lIter)->m_pre_timestamp > 1e9) {

					delete *m_lIter;
					m_lObstacle.erase(m_lIter++);
					continue;
				}

				float tempx=0,tempy=0,tempyaw=0;
				(*m_lIter)->GetLocation(_ttimestamp, tempx, tempy, tempyaw);
				float distx = lx - tempx;
				float disty = ly - tempy;
				float tempdistance = sqrt(distx*distx + disty*disty);
				if (tempdistance > 3) {
					std::cout << (*m_lIter)->m_id << " " << tempdistance << " ";
					std::cout << _ttimestamp << " " << (*m_lIter)->m_pre_timestamp << " " << m_qRadpoints.back().timestamp << " " << tempx << " " << objectPoints[index].x;
				}
				if (tempdistance < 3 && tempdistance < distance) {
					distance = tempdistance;
					obsid = (*m_lIter)->m_id;
					t_lIter = m_lIter;
				}
				m_lIter++;
			}

			std::cout << "#" << obsid;
			if (obsid == -1) {
				//add the new location to obstacle map
				Obstacle *tempobs = new Obstacle();
				tempobs->InitRadar(m_iTracknum, lclass_id, _ttimestamp,
				 	lx, ly, lz);
				m_lObstacle.push_back(tempobs);
				m_iTracknum++;

			}
			else {
				//update obstacles map
				(*t_lIter)->UpdateMeasureRadar(_ttimestamp, lx, ly, lz, ///////////////////////////////////////
					lr, lphi, lr_dot);
				std::cout << "." ;
			}
		}
		std::cout << "#" << std::endl;

	};
	void Output(long t, mychallenge::MyPoseArray &PoseArray) {

		float dis=FLT_MAX;
		float outx =0,outy=0,outyaw=0,outo=0, outz=0, outl=0, outw=0,outh=0;
		int outid;
		long outt;
		for (m_lIter = m_lObstacle.begin(); m_lIter != m_lObstacle.end(); ) {
			if (t - (*m_lIter)->m_pre_timestamp > 15e8) {
				delete *m_lIter;
				m_lObstacle.erase(m_lIter++);
				continue;
			}
			if (t - (*m_lIter)->m_pre_timestamp > 10e8) {
				m_lIter++;
				continue;
			}
			if((*m_lIter)->m_trackcount < 5){
				m_lIter++;
				continue;
			}
			float tempx =0,tempy=0,tempyaw=0,tempo=0, tempz=0;
			(*m_lIter)->GetLocation(t, tempx, tempy, tempyaw);
			(*m_lIter)->GetOri(t, tempo);
			//(*m_lIter)->GetZ(t, tempz);
			tempz = (*m_lIter)->m_z;
			float l = 0.8, w = 0.8, h = 1.708;
			if((*m_lIter)->m_type == 1){
				l = (*m_lIter)->m_length;
				w = (*m_lIter)->m_width;
				h = (*m_lIter)->m_height;
			}
			// PoseArray.obs_id.push_back((*m_lIter)->m_id);
			// PoseArray.x.push_back(tempx);
			// PoseArray.y.push_back(tempy);
			// PoseArray.z.push_back(tempz);
			// PoseArray.orientation.push_back(tempyaw);
			// PoseArray.l.push_back(l);
			// PoseArray.w.push_back(w);
			// PoseArray.h.push_back(h);
			float tempdis=sqrt(tempx*tempx+tempy*tempy);
			if(tempdis<dis){
				dis = tempdis;
				outid = (*m_lIter)->m_id;
				outt = t;
				outx = tempx;
				outy = tempy;
				outz = tempz;
				outyaw = tempyaw;
				outo = tempo;
				outl = l;
				outw = w;
				outh = h;
			}
			// out<<(*m_lIter)->m_id<<" "<<t<<" "<<tempx<<" "
			// 	<<tempy<<" "<<(*m_lIter)->m_z<<" "<<tempyaw<<" "<<tempo<<" "//(*m_lIter)->m_orientation<<" "
			// 	<<l<<" "<<w<<" "<<h<<std::endl;
			m_lIter++;
		}
		if(dis!=FLT_MAX){
			//The rule sadi: if there are mant obstacls, output the closest one 
			out<<outid<<" "<<outt<<" "<<outx<<" "
			<<outy<<" "<<outz<<" "<<outyaw<<" "<<outo<<" "//(*m_lIter)->m_orientation<<" "
			<<outl<<" "<<outw<<" "<<outh<<std::endl;
			PoseArray.obs_id.push_back(outid);
			PoseArray.x.push_back(outx);
			PoseArray.y.push_back(outy);
			PoseArray.z.push_back(outz);
			PoseArray.orientation.push_back(outyaw);
			PoseArray.l.push_back(outl);
			PoseArray.w.push_back(outw);
			PoseArray.h.push_back(outh);
		}
	};

};
