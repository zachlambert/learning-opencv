#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main()
{
    // TODO: Instal intel realse sdk, use to access T265
    cv::VideoCapture video;
    video.open(1);
    if(!video.isOpened()){
        std::cout << "Could not open video capture." << std::endl;
        return 1;
    }
    cv::Mat frame;
    while(true){
        video >> frame;
        cv::imshow("Frame", frame);
        char c = (char)cv::waitKey(25);
        if(c==27) break;
    }
    return 0;
}
