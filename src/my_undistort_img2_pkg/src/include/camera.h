#pragma once
#include "macros.h"
#include <iostream>
#include <fstream>
#include <ctime>

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class Camera{

    cv::Point3d coordinates;
public:
    std::string namePosition;
    std::string rstpAddress;

    int resolutionX;
    int resolutionY;

    bool isDistortionCalibrated;
    bool isPerspTransCalibrated;

    cv::Mat imageLive;
    cv::Mat imageLiveUndistorted;
    cv::Mat imageLiveTranformedPerspective;

    cv::VideoWriter video;

    bool initializeCamera(std::string _namePosition, std::string _rstpAddress, std::string _pathDistortionMatrix, std::string _pathPerpTransMatrix);
    clock_t checkFPS();
    cv::Mat getImage( bool checkFPS);
    cv::Mat undistort();
    cv::Mat transformPerspective();

    void record(float fps);
    void showView(std::string& rstpAddress, cv::Mat& image);

    Camera(){
    }
private:
    //Calibration Parameter
    std::vector<cv::Mat> instrExtrMatrix;
    cv::Mat perspTransParameter;

    //FPS Measurement
    clock_t fps = 0;
    std::chrono::high_resolution_clock::time_point t1 , t2;
    std::chrono::duration<double> time_span;
    int framesCaptured= 0;

    //Recording
    bool isRecording= false;
    cv::VideoCapture videoCapture;

    bool readIntrinsicExtrinsic(std::string filePath);
    bool readPerspTransParameter(std::string filePath);

};


