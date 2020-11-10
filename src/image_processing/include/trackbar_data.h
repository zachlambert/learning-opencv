#ifndef IMAGE_PROCESSING_TRACKBAR_DATA_H
#define IMAGE_PROCESSING_TRACKBAR_DATA_H

#include <opencv2/core.hpp>

class TrackbarData{
private:
    typedef void (*filterFunction_t)(double, const cv::Mat&, cv::Mat&);
    double level;
    const int max_value;
    const cv::Mat* src;
    cv::Mat* dst;
    filterFunction_t filterFunction;
public:
    TrackbarData(int max_value, const cv::Mat& src, cv::Mat& dst,
                 filterFunction_t filterFunction):
        level(0), max_value(max_value), src(&src), dst(&dst),
        filterFunction(filterFunction) {}

    void setSrc(const cv::Mat& src){ this->src = &src; }
    void setFilterFunction(filterFunction_t filterFunction){
        this->filterFunction  = filterFunction; }
    void applyFilter(int slider_value);
    const cv::Mat& getDst()const{ return *dst; }
    int getMaxValue()const{ return max_value; }
};

void createTrackbarWindow(TrackbarData* trackbarData);

#endif
