#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "trackbar_data.h"

void applyHomogeneousBlur(double level, const cv::Mat& src, cv::Mat& dst){
    int kernel_size = 1 + 2*(int)(level*20);
    cv::blur(src, dst, cv::Size(kernel_size, kernel_size), cv::Point(-1,-1));
}

void applyGaussianBlur(double level, const cv::Mat& src, cv::Mat& dst){
    int kernel_size = 1 + 2*(int)(level*20);
    cv::GaussianBlur(src, dst, cv::Size(kernel_size, kernel_size), 0, 0);
}

void applyMedianBlur(double level, const cv::Mat& src, cv::Mat& dst){
    int kernel_size = 1 + 2*(int)(level*20);
    cv::medianBlur(src, dst, kernel_size);
}

void applyBilateralBlur(double level, const cv::Mat& src, cv::Mat& dst){
    int kernel_size = 1 + 2*(int)(level*20);
    cv::bilateralFilter(src, dst, kernel_size, kernel_size*2, kernel_size/2);
}

void applyErosion(double level, const cv::Mat& src, cv::Mat& dst){
    int size = 1 + (int)(level*5);
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2*size + 1, 2*size+1),
        cv::Point(size, size));
    cv::erode(src, dst, element);
}

void applyDilation(double level, const cv::Mat& src, cv::Mat& dst){
    int size = 1 + (int)(level*5);
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2*size + 1, 2*size+1),
        cv::Point(size, size));
    cv::dilate(src, dst, element);
}

int main(int argc, char** argv){
    cv::Mat dst;
    cv::samples::addSamplesDataSearchPath("data");
    cv::Mat src = cv::imread(cv::samples::findFile("lena.jpg"));
    cv::Mat src2 = cv::imread(cv::samples::findFile("LinuxLogo.jpg"));

    // Blur filters

    TrackbarData trackbarData(100, src, dst, applyHomogeneousBlur);
    createTrackbarWindow(&trackbarData);
    cv::waitKey(0);

    trackbarData.setFilterFunction(applyGaussianBlur);
    cv::waitKey(0);

    trackbarData.setFilterFunction(applyMedianBlur);
    cv::waitKey(0);

    trackbarData.setFilterFunction(applyBilateralBlur);
    cv::waitKey(0);

    // Dilation and erosion on linux logo

    trackbarData.setSrc(src2);
    trackbarData.setFilterFunction(applyErosion);
    cv::waitKey(0);

    trackbarData.setFilterFunction(applyDilation);
    cv::waitKey(0);

    // Dilation and erosion on lena

    trackbarData.setSrc(src);
    trackbarData.setFilterFunction(applyErosion);
    cv::waitKey(0);

    trackbarData.setFilterFunction(applyDilation);
    cv::waitKey(0);

    return 0;
}
