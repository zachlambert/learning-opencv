#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main(int argc, char** argv){
    cv::VideoCapture video;
    video.open(0);
    if(!video.isOpened()){
        std::cout << "Could not open video capture." << std::endl;
        return 1;
    }
    cv::Mat frame, gray, temp;
    video >> frame; // Use for getting image size
    cv::Mat window = cv::Mat::ones(3, 3, CV_8U);
    int op=0;

    auto sift = cv::SIFT::create();
    auto fast = cv::FastFeatureDetector::create();
    auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;

    while(true){
        video >> frame; // Overloaded >> operator, reads current frame into a mat
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1);
        if(frame.empty()){
            break;
        }
        switch(op){
            case 0:
                // Base image
                cv::imshow("Frame", gray);
                break;
            case 1:
                sift->detect(gray, keypoints);
                cv::drawKeypoints(gray, keypoints, temp);
                cv::imshow("Frame", temp);
            case 2:
                fast->detect(gray, keypoints);
                cv::drawKeypoints(gray, keypoints, temp);
                cv::imshow("Frame", temp);
            case 3:
                // X component of the gradient
                orb->detect(gray, keypoints);
                cv::drawKeypoints(gray, keypoints, temp);
                cv::imshow("Frame", temp);
                break;
            default:
                return 0;
        }
        char c = (char)cv::waitKey(25);
        if(c==27) op++;
    }
}
