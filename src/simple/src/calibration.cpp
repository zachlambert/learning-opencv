#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/calib3d/calib3d_c.h>
#include <iostream>

int main(int argc, char** argv){
    cv::VideoCapture video;
    video.open(0);
    if(!video.isOpened()){
        std::cout << "Could not open video capture." << std::endl;
        return 1;
    }
    cv::Mat frame, gray;
    cv::Mat corners;
    bool result = false;

    cv::Size pattern_size(4, 4); // Interior number of corners
    int chessboard_flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;

    char c = 0;
    while(c != 27 && !result){
        video >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        result = cv::findChessboardCorners(gray, pattern_size, corners, chessboard_flags);
        cv::imshow("Frame", frame);
        c = (char)cv::waitKey(25);
    }

    if (!result) return 1;

    cv::TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
    cv::drawChessboardCorners(frame, pattern_size, corners, result);
    cv::imshow("Result", frame);

    std::cout << corners << std::endl;

    std::vector<std::vector<cv::Point2f>> image_points;
    image_points.push_back(std::vector<cv::Point2f>());
    for (std::size_t i = 0; i < corners.rows; i++) {
        double *ptr = corners.ptr<double>(i);
        image_points[0].push_back(cv::Point2f(ptr[0], ptr[1]));
    }

    std::vector<std::vector<cv::Point3f>> object_points;

    object_points.push_back(std::vector<cv::Point3f>());
    for (int x = 0; x < pattern_size.width; x++) {
        for (int y = 0; y < pattern_size.height; y++) {
            object_points[0].push_back(cv::Point3f(0.03f*x, 0.03f*y, 0));
        }
    }

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(
        object_points, image_points, cv::Size(frame.cols, frame.rows), camera_matrix,
        dist_coeffs, rvecs, tvecs);
    std::cout << "Average reprojection error: " << rms << std::endl;
    std::cout << "Camera projection matrix:" << std::endl << camera_matrix << std::endl;
    // Getting bad rms with this. Probably need to do something extra.


    c = 0;
    while (c!=27) {
        c = (char)cv::waitKey(25);
    }
}
