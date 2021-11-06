#ifndef MASKDETECTION_H
#define MASKDETECTION_H

#include<opencv2/opencv.hpp>
#include "Yolov5.h"
class Mask_Detection{
    public:
        Mask_Detection();
        void process();

    private:
        cv::Mat srcImage;
        cv::VideoCapture cap;
    };
#endif //__SERIAL_PORT_THREAD_H
