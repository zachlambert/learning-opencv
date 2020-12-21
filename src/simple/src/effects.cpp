#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

void weightedSquareDifference(const cv::Mat& src, cv::Mat& dest, const cv::Mat& window, cv::Point anchor = cv::Point(-1, -1)){
    CV_Assert(src.depth() == CV_8U);
    if(anchor==cv::Point(-1, -1)){
        anchor = cv::Point((window.cols-1)/2, (window.rows-1)/2);
    }
    for(int y=0; y<src.rows; y++){
        for(int x=0; x<src.cols; x++){
            int sum = 0;
            for(int v=0; v<window.rows; v++){
                for(int u=0; u<window.cols; u++){
                    if(y+v-anchor.y>0 && y+v-anchor.y<src.rows &&
                            x+u-anchor.x>0 && x+u-anchor.x<src.cols){
                        sum += (int)window.ptr<uchar>(v)[u] *
                                pow(src.ptr<uchar>(y+v-anchor.y)[x+u-anchor.x]
                                    - src.ptr<uchar>(y)[x], 2);
                    }
                }
            }
            dest.ptr<uchar>(y)[x] = (uchar)sum;
        }
    }
}

int main(int argc, char** argv){
    cv::VideoCapture video;
    video.open(0);
    if(!video.isOpened()){
        std::cout << "Could not open video capture." << std::endl;
        return 1;
    }
    cv::Mat frame, output;
    video >> frame; // Use for getting image size
    cv::Mat gray;
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Mat kernel;
    cv::Mat window = cv::Mat::ones(3, 3, CV_8U);
    int op=0;
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
                // X component of the gradient
                cv::Sobel(gray, grad_x, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
                cv::imshow("Frame", grad_x);
                break;
            case 2:
                // Y component of the gradient
                cv::Sobel(gray, grad_y, CV_8U, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
                cv::imshow("Frame", grad_y);
                break;
            case 3:
                // Magnitude of the gradient
                // src, dst, depth, dx, dy, kernel size, scale, delta, border type
                cv::Sobel(gray, grad_x, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
                cv::Sobel(gray, grad_y, CV_8U, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
                grad_x.convertTo(grad_x, CV_32F);
                grad_y.convertTo(grad_y, CV_32F);
                cv::magnitude(grad_x, grad_y, grad_mag);
                grad_mag.convertTo(grad_mag, CV_8U);
                cv::normalize(grad_mag, grad_mag, 0, 255, cv::NORM_MINMAX);
                cv::imshow("Frame", grad_mag);
                break;
            case 4:
                // What happens if we convolve (filter) the image with
                // various kernels?
                // Vertical step kernel -> Large for horizontal edges
                kernel = (cv::Mat_<char>(3,3) << 5,5,5, 0,0,0, -5,-5,-5);
                cv::filter2D(gray, output, -1, kernel);
                cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
                cv::imshow("Frame", output);
                break;
            case 5:
                // Horizontal step kernel -> Large for vertical edges
                kernel = (cv::Mat_<char>(3,3) << 5,0,-5, 5,0,-5, 5,0,-5);
                cv::filter2D(gray, output, -1, kernel);
                cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
                cv::imshow("Frame", output);
                break;
            case 6:
                // An edge detection kernel which detects edges
                // in all directions
                kernel = (cv::Mat_<char>(3,3) << 1,0,-1, 0,0,0, -1,0,1);
                cv::filter2D(gray, output, -1, kernel);
                cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
                cv::imshow("Frame", output);
                break;
            case 7:
                // A better edge detection kernel
                kernel = (cv::Mat_<char>(3,3) << 0,1,0, 1,-4,1, 0,1,0);
                cv::filter2D(gray, output, -1, kernel);
                cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
                cv::imshow("Frame", output);
                break;
            case 8:
                // An even better edge detection kernel
                kernel = (cv::Mat_<char>(3,3) << -1,-1,-1, -1,8,-1, -1,-1,-1);
                cv::filter2D(gray, output, -1, kernel);
                cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
                cv::imshow("Frame", output);
                break;
            case 9:
                // Rectangular window
                weightedSquareDifference(gray, output, window);
                cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
                cv::imshow("Frame", output);
                break;
            case 10:
                // Find the gradient of the above, which simplfies to a
                // linear operation
            default:
                return 0;
        }
        char c = (char)cv::waitKey(25);
        if(c==27) op++;
    }
    return 0;
}
