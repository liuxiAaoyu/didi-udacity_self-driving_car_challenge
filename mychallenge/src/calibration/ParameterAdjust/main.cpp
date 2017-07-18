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
#include<fstream>
#include<opencv2/opencv.hpp>

using namespace std;


int main(){
    string s;
    ifstream cin("1492886950.073595000.txt");
    vector<cv::Point3f> objectPoints;
    while(cin.eof()!=1){
        double tx,ty,tz,t;
        cin>>tx>>ty>>tz>>t>>t>>t;
        if(tx>1)
            objectPoints.push_back(cv::Point3f(tx,ty,tz));
        //cout<<tx<<" "<<ty<<" "<<tz<<" "<<endl;
    }
    cv::Mat img;
   //img = cv::imread("1492886949.503365157.jpg");



    //1.20514 -1.22953 1.17914 1399.62 1528.65

    //cv::Mat m_cam = (cv::Mat_<double>(3,3) << 1384.621562, 0.000000, 625.888005,0.000000, 1393.652271, 559.626310,0,0,1);
    double fx=1394.62 , fy =1568.65, cx = 625.888005, cy = 559.626310;//1394.62 1568.65
    cv::Mat m_dist = (cv::Mat_<double>(5,1) << 0, 0, 0, 0, 0);// -0.152089, 0.270168, 0.003143, -0.005640, 0.000000);
    double rx=1.20914 ,ry=-1.22553 ,rz=1.18114  ;//1.20914 -1.22553 1.18114
    //   1.18913633 -1.23253128 1.2014496  //1.20513633 -1.22253128 1.17414496  //1.18513633 -1.17253128 1.19414496
    double tx=-0.04032959,ty=-0.40270426,tz=-0.47190471;
    //cv::Mat m_tvecs = (cv::Mat_<double>(3,1) << -0.04032959, -0.40270426, -0.47190471);
    cv::Mat img_projected;
    img = cv::imread("1492886950.086670850.jpg");
    while(1){
        int key = cv::waitKey(0);
        switch (key)  {
            case 'q':
                rx += 0.001;
                break;
            case 'w':
                rx -= 0.001;
                break;
            case 'a':
                ry += 0.001;
                break;
            case 's':
                ry -= 0.001;
                break;
            case 'z':
                rz += 0.001;
                break;
            case 'x':
                rz -= 0.001;
                break;

            case 'e':
                fx += 5;
                break;
            case 'r':
                fx -= 5;
                break;
            case 'd':
                fy += 5;
                break;
            case 'f':
                fy -= 5;
                break;

        }
        std::vector<cv::Point2f> imagePoints;
        cv::Mat m_cam = (cv::Mat_<double>(3,3) << fx, 0.000000, cx,0.000000, fy, cy,0,0,1);
        cv::Mat m_rvecs = (cv::Mat_<double>(3,1) << rx, ry, rz);
        cv::Mat m_tvecs = (cv::Mat_<double>(3,1) << tx, ty, tz);
        cv::projectPoints(objectPoints, m_rvecs, m_tvecs, m_cam, m_dist, imagePoints);

        img.copyTo(img_projected);
        for (int i =0; i<imagePoints.size();i++){
          int x=(int)imagePoints[i].x;
          int y=(int)imagePoints[i].y;
          if(0<=x&&x<1368&&y>=0&&y<1096){
            img_projected.at<cv::Vec3b>(y,x)[2]=255;
            img_projected.at<cv::Vec3b>(y,x)[0]=255;
            img_projected.at<cv::Vec3b>(y,x)[1]=255;
          }
        }
        cout<<rx<<" "<<ry<<" "<<rz<<" "<<endl;
        cout<<fx<<" "<<fy<<" "<<endl;
        //imwrite("/home/xiaoyu/round.jpg",img_projected);
        //cv::pyrDown(img_projected,img_projected);
        imshow("img",img_projected);
    }

    return 1;
}
