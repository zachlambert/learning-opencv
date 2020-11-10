#include "trackbar_data.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


void TrackbarData::applyFilter(int slider_value){
    level = (double)slider_value/max_value;
    filterFunction(level, *src, *dst);
}

static void trackbarCallback(int slider_value, void* trackbarData_vp){
    TrackbarData& data = *(TrackbarData*)(trackbarData_vp);
    data.applyFilter(slider_value);
    cv::imshow("Image Processing", data.getDst());
}

// I think you need to use a pointer to the base class TrackbarData in order
// for it to dynamically cast the object and allow polymorphism.
void createTrackbarWindow(TrackbarData* trackbarData){
    int slider_value = 0;
    cv::namedWindow("Image Processing", cv::WINDOW_AUTOSIZE);
    char trackbarName[50];

    std::sprintf(trackbarName, "Level x %d", trackbarData->getMaxValue());
    cv::createTrackbar(trackbarName, "Image Processing", &slider_value,
                       trackbarData->getMaxValue(), trackbarCallback, trackbarData);

    trackbarCallback(slider_value, trackbarData);
}
