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
        cv::filter2D(gray, dsdx, CV_32F, kernel_dsdx);
        cv::filter2D(gray, dsdy, CV_32F, kernel_dsdy);
        // Can also do this directly with the cv::Sobel function

        cv::Mat ds_mag(dsdx.rows, dsdx.cols, CV_32F);
        cv::magnitude(dsdx, dsdy, ds_mag);

        double max_mag;
        cv::minMaxLoc(ds_mag, 0, &max_mag);

        float mag_threshold = max_mag*0.2;


        // Now use non-maximum suppression to only retain pixels for
        // which ds_mag is a local maxima along the edge direction
        cv::Mat edges = cv::Mat(ds_mag.rows, ds_mag.cols, CV_8UC1);

        for (int i = 0; i < ds_mag.rows; i++) {
            for (int j = 0; j < ds_mag.cols; j++) {
                // Image only has one channel
                float mag = ds_mag.ptr<float>(i)[j];
                float nx = dsdx.ptr<float>(i)[j]/mag;
                float ny = dsdy.ptr<float>(i)[j]/mag;
                int ni = ceil(nx);
                int nj = ceil(ny);
                float prev_mag = 0;
                if (i-ni >= 0 && i-ni < ds_mag.rows && j-nj >=0 && j-nj < ds_mag.cols) {
                    prev_mag = ds_mag.ptr<float>(i-ni)[j-nj];
                }
                float next_mag = 0;
                if (i+ni >= 0 && i+ni < ds_mag.rows && j+nj >=0 && j+nj < ds_mag.cols) {
                    prev_mag = ds_mag.ptr<float>(i+ni)[j+nj];
                }
                if (mag >= next_mag && mag >= prev_mag && mag > mag_threshold) {
                    edges.ptr<uchar>(i)[j] = (mag/max_mag)*255;
                } else {
                    edges.ptr<uchar>(i)[j] = 0;
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

class MarrHildrethDetectorCustom: public ImageOperation {
public:
    MarrHildrethDetectorCustom(int ksize, double sigma) {
        kernel = cv::getGaussianKernel(ksize, sigma);
        cv::Laplacian(kernel, kernel, -1);
    }

    void apply(const cv::Mat &gray) {
        cv::filter2D(gray, d2s, CV_16S, kernel);
        // Now identify zero crossings in laplacian of smoothed image
        output = cv::Mat(d2s.rows, d2s.cols, CV_8UC1);
        for (int i = 0; i < d2s.rows; i++) {
            for (int j = 0; j < d2s.cols; j++) {
                bool pos_sign = false;
                bool neg_sign = false;
                for (int di = -1; di < 2; di++) {
                    for (int dj = -1; dj < 2; dj++) {
                        if (i+di>=0 && i+di<d2s.rows && j+dj>=0 && j+dj<d2s.cols) {
                            if (d2s.ptr<short>(i+di)[j+dj] > 0) {
                                pos_sign = true;
                            } else if (d2s.ptr<short>(i+di)[j+dj] < 0) {
                                neg_sign = true;
                            }

                        }
                    }
                }
                if (pos_sign && neg_sign) {
                    output.ptr<uchar>(i)[j] = 255;
                } else {
                    output.ptr<uchar>(i)[j] = 0;
                }
            }
        }
    }
private:
    cv::Mat kernel;
    cv::Mat d2s;
};

int gaussian_ksize(double sigma) {
    int ksize = 2*ceil(3.7*sigma - 1) + 1;
    if (ksize > 31) ksize=31;
    return ksize;
}

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
        new SmoothOperation(gaussian_ksize(1), 1)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new SmoothOperation(gaussian_ksize(3), 3)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new CannyEdgeDetectorCustom(gaussian_ksize(0.5), 0.5)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new CannyEdgeDetectorCustom(gaussian_ksize(1), 1)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new CannyEdgeDetectorCustom(gaussian_ksize(3), 3)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new CannyEdgeDetectorCustom(gaussian_ksize(7), 7)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new MarrHildrethDetectorCustom(gaussian_ksize(4), 4)
    ));
    operations.push_back(std::unique_ptr<ImageOperation>(
        new MarrHildrethDetectorCustom(gaussian_ksize(2), 2)
    ));

    while(true){
        video >> frame; // Overloaded >> operator, reads current frame into a mat
        if(frame.empty()){
            break;
        }
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        cv::flip(gray, gray, 1);
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
