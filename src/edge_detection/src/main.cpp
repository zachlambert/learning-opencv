#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

class ImageOperation {
public:
    virtual void apply(const cv::Mat &gray)=0;
    const cv::Mat &get_output()const{ return output; }
protected:
    cv::Mat output;
};

class NoOperation: public ImageOperation {
    void apply(const cv::Mat &gray) {
        output = gray;
    }
};

class SmoothOperation: public ImageOperation {
public:
    SmoothOperation(int ksize, double sigma) {
        kernel = cv::getGaussianKernel(ksize, sigma);
    }
    void apply(const cv::Mat &gray) {
        cv::filter2D(gray, output, -1, kernel);
    }
private:
    cv::Mat kernel;
};

class CannyEdgeDetectorCustom: public ImageOperation {
public:
    CannyEdgeDetectorCustom(int ksize, double sigma) {
        cv::Mat temp = cv::getGaussianKernel(ksize, sigma);
        cv::getDerivKernels(kernel_dsdx, kernel_dsdy, 1, 1, ksize);
        cv::filter2D(kernel_dsdx, kernel_dsdx, -1, temp);
        cv::filter2D(kernel_dsdy, kernel_dsdy, -1, temp);

    }
    void apply(const cv::Mat &gray) {
        // Use depth of CV_16S in gradient channel, to avoid overflow
        // 16S = 16 bits per pixel, signed
        cv::filter2D(gray, dsdx, CV_16S, kernel_dsdx);
        cv::filter2D(gray, dsdy, CV_16S, kernel_dsdy);
        // Can also do this directly with the cv::Sobel function

        cv::Mat ds_mag(dsdx.rows, dsdx.cols, CV_64F);
        cv::magnitude(dsdx, dsdy, ds_mag);

        // Now use non-maximum suppression to only retain pixels for
        // which ds_mag is a local maxima along the edge direction
        cv::Mat edges = ds_mag.clone();
        for (int i = 0; i < ds_mag.rows; i++) {
            for (int j = 0; j < ds_mag.cols; j++) {
                // Image only has one channel
                double mag = ds_mag.ptr<uchar>(i)[j];
                double nx = dsdx.ptr<short>(i)[j]/mag;
                double ny = dsdy.ptr<short>(i)[j]/mag;
                int ni = ceil(nx);
                int nj = ceil(ny);
                double prev_mag = ds_mag.ptr<short>(i-ni)[j-nj];
                double next_mag = ds_mag.ptr<short>(i+ni)[j+nj];
                if (mag < next_mag || mag < prev_mag) {
                    edges.ptr<short>(i)[i] = 0;
                }
            }
        }
        output = edges;
    }
private:
    cv::Mat kernel_dsdx;
    cv::Mat kernel_dsdy;
    cv::Mat dsdx, dsdy, ds_mag;
};

int main()
{
    cv::VideoCapture video;
    video.open(0);
    if(!video.isOpened()){
        std::cout << "Could not open video capture." << std::endl;
        return 1;
    }

    cv::Mat frame, gray, output;
    video >> frame; // Use for getting image size

    std::vector<std::unique_ptr<ImageOperation>> operations;
    int op = 0;

    operations.push_back(std::unique_ptr<ImageOperation>(
        new NoOperation()
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new SmoothOperation(5, 1)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new SmoothOperation(20, 5)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new CannyEdgeDetectorCustom(5, 1)
    ));

    while(true){
        video >> frame; // Overloaded >> operator, reads current frame into a mat
        if(frame.empty()){
            break;
        }
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        operations[op]->apply(gray);
        cv::imshow("Frame", operations[op]->get_output());

        if((char)cv::waitKey(10) == 27) {
            op++;
            if (op == operations.size()) {
                break;
            }
        }
    }
    return 0;
}
