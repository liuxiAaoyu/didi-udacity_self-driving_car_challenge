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
#include<sstream>
#include"ukf.hpp"
#include"ekf.hpp"


#define IMG_SCORE_THRESHOLD 0.1
#define LIDAR_SCORE_THRESHOLD 0.1
#define IMG_HEIGHT 1096
#define IMG_WIDTH 1368
#define LIDAR_IMG_HEIGHT 600
#define LIDAR_IMG_WIDTH 600
//obstacle class
//use this class to tracking obstacle
class Obstacle {
public:
	Obstacle() {};
	~Obstacle() {};
	bool m_trackflag;
	int m_trackcount;
	int m_id;
	int m_type;
	int m_sensor;  // 1 lidar, 2 radar
	float m_x;
	float m_y;
	float m_z;
	float m_orientation;
	float m_length;
	float m_width;
	float m_height;
	float m_size_conf;
	float m_boxh;
	float m_boxw;

	UKF ukf_;
	EKF ekf_;
	EKF okf_;
	EKF zkf_;
    long m_pre_timestamp;
    //acceleration noise components
    float noise_o;
    float noise_z;
    float noise_ax;
    float noise_ay;


	void Init(int tracknum, int type, long tt, float tx, float ty, float tzmax, 
		float tzmin, float to, float tw, float th) {
		m_id = tracknum;
		m_trackcount = 0;
		m_trackflag = false;
		m_type = type;
		m_sensor = 1;
		inintKalman();
		if(m_type ==1 || m_type == 2){
        //set the acceleration and yaw noise components
			if(m_type == 2){
				ukf_.std_a_ = 1;
				ukf_.std_yawdd_ = 9;
				if(tzmin > -1.4 && tzmax- tzmin<1.4)
					tzmin = tzmax - 1.708;
			}

			noise_o = 2;
			noise_z = 1;
			//state covariance matrix P
			ukf_.P_ = Eigen::MatrixXd(5, 5);
			ukf_.P_ << 0.5, 0, 0, 0, 0,
					0, 0.5, 0, 0, 0,
					0, 0, 5, 0, 0,
					0, 0, 0, 3, 0,
					0, 0, 0, 0, 2;
			m_x = tx;
			m_y = ty;
			m_z = (tzmax+tzmin)/2;
			m_orientation = to;
			okf_.x_ << to, 0;
			zkf_.x_ << m_z, 0;
			m_boxh = th;
			m_boxw = tw;
			cal_wl(to, th, tw, tx, ty, m_length, m_width, m_size_conf);
			m_height = tzmax-tzmin;
			if(m_height < 1.5)
				m_height = 1.5;
			m_pre_timestamp = tt;
			ukf_.x_ << tx, ty, 0, 0, 0;
		}
		if(m_type == 2){
			if(m_type == 1){
				noise_ax = 15;
				noise_ay = 15;
			}
			if(m_type == 2){
				noise_ax = 5;
				noise_ay = 5;
			}
			noise_o = 2;
			ekf_.x_ = Eigen::VectorXd(4);

			//state covariance matrix P
			ekf_.P_ = Eigen::MatrixXd(4, 4);
			ekf_.P_ << 1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1000, 0,
					0, 0, 0, 1000;

			//measurement covariance
			ekf_.R_ = Eigen::MatrixXd(2, 2);
			ekf_.R_ << 0.25, 0,
					0, 0.25;

			//measurement matrix
			ekf_.H_ = Eigen::MatrixXd(2, 4);
			ekf_.H_ << 1, 0, 0, 0,
					0, 1, 0, 0;

			//the initial transition matrix F_
			ekf_.F_ = Eigen::MatrixXd(4, 4);
			ekf_.F_ << 1, 0, 1, 0,
					0, 1, 0, 1,
					0, 0, 1, 0,
					0, 0, 0, 1;

			m_x = tx;
			m_y = ty;
			if(tzmin > -1.4 && tzmax- tzmin<1.4)
				tzmin = tzmax - 1.708;
			m_z = (tzmax+tzmin)/2;
			m_orientation = to;
			okf_.x_ << to, 0;
			zkf_.x_ << m_z, 0;

			m_boxh = th;
			m_boxw = tw;

			cal_wl(to, th, tw, tx, ty, m_length, m_width, m_size_conf);
			m_height = tzmax-tzmin;
			if(m_height < 1.4)
				m_height = 1.4;

			m_pre_timestamp = tt;
			ekf_.x_ << tx, ty, 0, 0;
		}
		

		
	};

	void InitRadar(int tracknum, int type, long tt, float tx, float ty, float tz){
		m_id = tracknum;
		m_trackcount = 0;
		m_trackflag = false;
		m_type = type;
		m_sensor = 2;
		inintKalman();
		noise_o = 2;
		noise_z = 1;

		m_x = tx;
		m_y = ty;
		m_z = tz;

		m_height = 1.75;
		m_length = 5;
		m_width = 2;
		m_pre_timestamp = tt;
		ukf_.x_ << tx, ty, 0, 0, 0;
		ukf_.P_ = Eigen::MatrixXd(5, 5);
		ukf_.P_ << 2, 0, 0, 0, 0,
				0, 2, 0, 0, 0,
				0, 0, 5, 0, 0,
				0, 0, 0, 3, 0,
				0, 0, 0, 0, 2;
		okf_.x_ << 0, 0;
		zkf_.x_ << 0, 0;
	};

	void inintKalman(){
		
		okf_.x_ = Eigen::VectorXd(2);

        //state covariance matrix P
        okf_.P_ = Eigen::MatrixXd(2, 2);
        okf_.P_ << 1, 0,
				   0, 1000;

        //measurement covariance
        okf_.R_ = Eigen::MatrixXd(1, 1);
        okf_.R_ << 0.25;

        //measurement matrix
        okf_.H_ = Eigen::MatrixXd(1, 2);
        okf_.H_ << 1, 0;

        //the initial transition matrix F_
        okf_.F_ = Eigen::MatrixXd(2, 2);
        okf_.F_ << 1, 1,
				   0, 1;
		zkf_.x_ = Eigen::VectorXd(2);

        //state covariance matrix P
        zkf_.P_ = Eigen::MatrixXd(2, 2);
        zkf_.P_ << 0.5, 0,
				   0, 10;

        //measurement covariance
        zkf_.R_ = Eigen::MatrixXd(1, 1);
        zkf_.R_ << 0.3;

        //measurement matrix
        zkf_.H_ = Eigen::MatrixXd(1, 2);
        zkf_.H_ << 1, 0;

        //the initial transition matrix F_
        zkf_.F_ = Eigen::MatrixXd(2, 2);
        zkf_.F_ << 1, 1,
				   0, 1;

	};

	void cal_wl(float ori, float boxh, float boxw, float x, float y,
		float &l, float &w, float &conf){
		float o = 0;
		float pi = M_PI;//4*atan(1);
		o = ori;
		if (ori>pi/2)
			o = ori - pi/2;
		if (ori<-pi/2)
			o = pi + ori;
		if (-pi/2<=ori<0)
			o = ori*-1;
		float alpha = o;
		float beta = pi/2 - o;
		//when yaw is close to pi/4, the equation has infinity sloutions
		//I just give w an constant
		if(fabs(alpha-pi/4)<0.13){
			w = 1.85;
			float m = (boxw+boxh)/20;
			l = m - w*cos(pi/4);
		}
		else{
			float ca = cos(alpha);
			float cb = cos(beta);
			l = fabs( (boxh*ca - boxw*cb)/(ca*ca-cb*cb)*0.1 );
			w = fabs( (boxw*ca - boxh*cb)/(cb*cb-ca*ca)*0.1 );
		}
		//space priori, obs car too far or too close from capture car has a low confidence
		float ux[2]={7,-7},uy[2]={3,-3};
		float mu = 100;
		if(abs(x-ux[0])<abs(x-ux[1])){
			if( abs(y-uy[0])<abs(y-uy[1]) )
				conf = exp(-(x-ux[0])*(x-ux[0])/(2*mu)  - (y-uy[0])*(y-uy[0])/(2*mu) );
			else
				conf = exp(-(x-ux[0])*(x-ux[0])/(2*mu) - (y-uy[1])*(y-uy[1])/(2*mu) );
		}
		else{
			if( abs(y-uy[0])<abs(y-uy[1]) )
				conf = exp(-(x-ux[1])*(x-ux[1])/(2*mu)  - (y-uy[0])*(y-uy[0])/(2*mu) );
			else
				conf = exp(-(x-ux[1])*(x-ux[1])/(2*mu) - (y-uy[1])*(y-uy[1])/(2*mu) );
		}
		//orientation priori, obs car orientation is pi/4 has a  low confidence
		float muratio = pi*15/180;
		if(o<pi/4)
			conf *= exp(-(o*o)/(2*muratio*muratio));
		else
			conf *= exp(-(o-pi/2)/(2*muratio*muratio));
		//car size priori, car size always in range (1.65-2.05)x(4.35-5) meters
		conf *=exp(-(l-4.65)*(l-4.65)/(2*0.15*0.15) - (w-1.85)*(w-1.85)/(2*0.1*0.1) );
	};

	void GetLocation(long t,float &tx, float &ty, float &to) {
		if (m_trackcount < 5){
			tx = m_x;
			ty = m_y;
			to = m_orientation;
			return ;
		}
		float dt = (t - m_pre_timestamp) / 1000000000.0;
		if(dt == 0){
			tx = m_x;
			ty = m_y;
			to = m_orientation;
			return;
		}
		if(m_type == 1)
			ukf_.Prediction(dt, m_sensor, m_x, m_y, tx,ty,to);
		if(m_type == 2){
			float dt_2 = dt * dt;
			float dt_3 = dt_2 * dt;
			float dt_4 = dt_3 * dt;
		
			//Modify the F matrix so that the time is integrated
			ekf_.F_(0, 2) = dt;
			ekf_.F_(1, 3) = dt;
			Eigen::VectorXd t_x_; 
			t_x_ = ekf_.x_;
			t_x_(0) = m_x;
			t_x_(1) = m_y; 
			t_x_ = ekf_.F_ * t_x_;
			//std::cout<<t_x_;

			tx = t_x_(0);
			ty = t_x_(1);
			to = m_orientation;
		}

		std::cout << "&"<<dt;

		return ;
	};

	//void getValue(KalmanFilter kf, long t, float &to, float oo){
	void GetOri(long t, float &to){
		if (m_trackcount < 5){
			to = m_orientation;
			return ;
		}
		// if (t - m_pre_timestamp < 5e8){
		// 	tx = m_x;
		// 	ty = m_y;
		// 	tz = m_z;
		// 	return ;
		// }
		float dt = (t - m_pre_timestamp) / 1000000000.0;
		float dt_2 = dt * dt;
        float dt_3 = dt_2 * dt;
        float dt_4 = dt_3 * dt;
    
        //Modify the F matrix so that the time is integrated
        okf_.F_(0, 1) = dt;
		Eigen::VectorXd t_x_; 
		t_x_ = okf_.x_;
        t_x_ = okf_.F_ * t_x_;
		//std::cout<<t_x_;

		to = t_x_(0);

		return ;
	};

	void GetZ(long t, float &tz){
		if (m_trackcount < 5){
			tz = m_z;
			return ;
		}
		// if (t - m_pre_timestamp < 5e8){
		// 	tx = m_x;
		// 	ty = m_y;
		// 	tz = m_z;
		// 	return ;
		// }
		float dt = (t - m_pre_timestamp) / 1000000000.0;
		float dt_2 = dt * dt;
        float dt_3 = dt_2 * dt;
        float dt_4 = dt_3 * dt;
    
        //Modify the F matrix so that the time is integrated
        zkf_.F_(0, 1) = dt;
		Eigen::VectorXd t_x_; 
		t_x_ = zkf_.x_;
		t_x_(0) = m_z;
        t_x_ = zkf_.F_ * t_x_;
		//std::cout<<t_x_;

		tz = t_x_(0);

		return ;
	};

	void UpdateMeasureLidar(long tt, float tx, float ty, float tzmax, 
		float tzmin, float to, float tw, float th) {
		float dt = (tt - m_pre_timestamp) / 1000000000.0;
		std::cout << "*";
		if(dt == 0)
			return ;
		m_trackcount++;
		if (m_trackcount < 5) {
			// if (dt>1.5e8){
			// 	m_trackcount = -1;
			// 	return;
			// }
		}
		m_sensor = 1;
		std::cout << "*";
		if(m_type == 1){
			ukf_.Prediction(dt);
			Eigen::VectorXd measurements =  Eigen::VectorXd(2);
			measurements << tx,ty;
			//measurement update
			ukf_.Update(measurements);

			float dt_2 = dt * dt;
			float dt_3 = dt_2 * dt;
			float dt_4 = dt_3 * dt;
			okf_.F_(0, 1) = dt;
			okf_.Q_ = Eigen::MatrixXd(2, 2);
			okf_.Q_ <<  dt_4/4*noise_o,  dt_3/2*noise_o,
						0, dt_2*noise_o;
			
			//predict
			okf_.Predict();
			Eigen::VectorXd omeasurements =  Eigen::VectorXd(1);
			omeasurements << to;
			//measurement update
			okf_.Update(omeasurements);
			//std::cout<<"o"<<to<<" "<<okf_.x_(0);

			zkf_.F_(0, 1) = dt;
			zkf_.Q_ = Eigen::MatrixXd(2, 2);
			zkf_.Q_ <<  dt_4/4*noise_z,  dt_3/2*noise_z,
						0, dt_2*noise_z;
			
			if(m_type == 2){
				if(tzmin > -1.4 && tzmax- tzmin<1.4)
					tzmin = tzmax - 1.708;
			}
			
			//predict
			zkf_.Predict();
			Eigen::VectorXd zmeasurements =  Eigen::VectorXd(1);
			zmeasurements << (tzmax+tzmin)/2;
			//measurement update
			zkf_.Update(zmeasurements);
			

			m_x = tx;
			m_y = ty;
			m_z = (tzmax+tzmin)/2;
			m_orientation = to;
			m_height = tzmax-tzmin;
			if(m_height < 1.5)
				m_height = 1.5;

			m_boxh = th;
			m_boxw = tw;
			float tobsl=0,tobsw=0,tconf=0;
			cal_wl(to, th, tw, m_x, m_y, tobsl, tobsw, tconf);
			if(tconf>m_size_conf){
				m_length = tobsl;
				m_width = tobsw;
				m_size_conf = tconf;
			}
		}
		if(m_type == 2){
			float dt_2 = dt * dt;
			float dt_3 = dt_2 * dt;
			float dt_4 = dt_3 * dt;
		
			//Modify the F matrix so that the time is integrated
			ekf_.F_(0, 2) = dt;
			ekf_.F_(1, 3) = dt;
		
			//set the process covariance matrix Q
			ekf_.Q_ = Eigen::MatrixXd(4, 4);
			ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
					0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
					dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
					0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
			
			//predict
			ekf_.Predict();
			Eigen::VectorXd measurements =  Eigen::VectorXd(2);
			measurements << tx,ty;
			//measurement update
			ekf_.Update(measurements);

			okf_.F_(0, 1) = dt;
			okf_.Q_ = Eigen::MatrixXd(2, 2);
			okf_.Q_ <<  dt_4/4*noise_o,  dt_3/2*noise_o,
						0, dt_2*noise_o;
			//predict
			okf_.Predict();
			Eigen::VectorXd omeasurements =  Eigen::VectorXd(1);
			omeasurements << to;
			//measurement update
			okf_.Update(omeasurements);
			//std::cout<<"o"<<to<<" "<<okf_.x_(0);

			zkf_.F_(0, 1) = dt;
			zkf_.Q_ = Eigen::MatrixXd(2, 2);
			zkf_.Q_ <<  dt_4/4*noise_z,  dt_3/2*noise_z,
						0, dt_2*noise_z;
			
			if(m_type == 2){
				if(tzmin > -1.4 && tzmax- tzmin<1.4)
					tzmin = tzmax - 1.708;
			}			
			//predict
			zkf_.Predict();
			Eigen::VectorXd zmeasurements =  Eigen::VectorXd(1);
			zmeasurements << (tzmax+tzmin)/2;
			//measurement update
			zkf_.Update(zmeasurements);

			m_x = tx;
			m_y = ty;
			m_z = (tzmax+tzmin)/2;
			m_orientation = to;
			m_height = tzmax-tzmin;
			if(m_height < 1.4)
				m_height = 1.4;

			m_boxh = th;
			m_boxw = tw;
			float tobsl=0,tobsw=0,tconf=0;
			cal_wl(to, th, tw, m_x, m_y, tobsl, tobsw, tconf);
			if(tconf>m_size_conf){
				m_length = tobsl;
				m_width = tobsw;
				m_size_conf = tconf;
			}
		}
		
		m_pre_timestamp = tt;


		std::cout<<" "<<tx<<" "<<ty<<" "<<ukf_.x_.transpose()<<std::endl;
		//std::cout<<m_id<<" "<<m_pre_timestamp<<" "<<m_x<<" "<<m_y<<" "<<m_z<<" "<<m_orientation<<" "<<m_length<<" "<<m_width<<" "<<m_height<<" \n";//<<std::endl;
		//out<<m_id<<std::endl;//<<" "<<m_pre_timestamp<<" "<<tx<<" "<<ty<<" "<<m_z<<" "<<to<<" "<<m_length<<" "<<m_width<<" "//<<m_height<<std::endl;
		return;
	};

	void UpdateMeasureRadar(long tt, float tx, float ty, float tz, 
		float tr, float tphi, float tr_dot) {
		float dt = (tt - m_pre_timestamp) / 1000000000.0;
		std::cout << "*";
		if(dt == 0)
			return ;
		m_trackcount++;
		if (m_trackcount < 5) {
			// if (dt>1.5e8){
			// 	m_trackcount = -1;
			// 	return;
			// }
		}
		
		ukf_.Prediction(dt);
		Eigen::VectorXd measurements =  Eigen::VectorXd(3);
        measurements << tr+1,tphi,tr_dot;
		//measurement update
		std::cout << "*";
		ukf_.UpdateRadar(measurements);

		//orientation and z 

		m_x = tx;
		m_y = ty;
		m_z = tz;
		m_sensor = 2;
		m_pre_timestamp = tt;
		std::cout<<std::endl;
		std::cout<<" "<<tx<<" "<<ty<<" "<<ukf_.x_.transpose();
		//std::cout<<m_id<<" "<<m_pre_timestamp<<" "<<m_x<<" "<<m_y<<" "<<m_z<<" "<<m_orientation<<" "<<m_length<<" "<<m_width<<" "<<m_height<<" \n";//<<std::endl;
		//out<<m_id<<std::endl;//<<" "<<m_pre_timestamp<<" "<<tx<<" "<<ty<<" "<<m_z<<" "<<to<<" "<<m_length<<" "<<m_width<<" "//<<m_height<<std::endl;
		return;
	};


};