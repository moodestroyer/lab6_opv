#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;

Mat convert_to_gray(Mat& img) {

	Mat gray_img(img.rows, img.cols, CV_8UC1);

#pragma omp parallel for
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			int gray_value = 0.29 * pixel[2] + 0.58 * pixel[1] + 0.11 * pixel[0];
			gray_img.at<uchar>(i, j) = gray_value;
		}
	}
	return gray_img;
}


Mat convert_to_negative(Mat& img) {

	Mat negative_img(img.rows, img.cols, CV_8UC3);

#pragma omp parallel for
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {

			Vec3b pixel = img.at<Vec3b>(i, j);
			int blue = pixel[0];
			int green = pixel[1];
			int red = pixel[2];

			int neg_b = 255 - blue;
			int neg_g = 255 - green;
			int neg_r = 255 - red;

			negative_img.at<Vec3b>(i, j) = Vec3b(neg_b, neg_g, neg_r);
		}
	}

	return negative_img;
}

Mat convert_to_sepia(Mat& img) {

	Mat sepia_img(img.rows, img.cols, CV_8UC3);

#pragma omp parallel for
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {

			Vec3b pixel = img.at<Vec3b>(i, j);
			int blue = pixel[0];
			int green = pixel[1];
			int red = pixel[2];

			int sep_b = (int)(0.272 * red + 0.534 * green + 0.131 * blue);
			int sep_g = (int)(0.349 * red + 0.686 * green + 0.168 * blue);
			int sep_r = (int)(0.393 * red + 0.769 * green + 0.189 * blue);

			sep_b = std::min(255, std::max(0, sep_b));
			sep_g = std::min(255, std::max(0, sep_g));
			sep_r = std::min(255, std::max(0, sep_r));

			sepia_img.at<Vec3b>(i, j) = Vec3b(sep_b, sep_g, sep_r);
		}
	}

	return sepia_img;
}

Mat convert_to_contour(Mat& img) {

	Mat contour_img(img.rows, img.cols, CV_8UC1);
	Mat grayImg;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);

#pragma omp parallel for
	for (int i = 1; i < grayImg.rows - 1; i++) {
		for (int j = 1; j < grayImg.cols - 1; j++) {
			float gx = grayImg.at<uchar>(i + 1, j + 1) + 2 * grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i - 1, j + 1) - grayImg.at<uchar>(i + 1, j - 1) - 2 * grayImg.at<uchar>(i, j - 1) - grayImg.at<uchar>(i - 1, j - 1);
			float gy = grayImg.at<uchar>(i + 1, j + 1) + 2 * grayImg.at<uchar>(i + 1, j) + grayImg.at<uchar>(i + 1, j - 1) - grayImg.at<uchar>(i - 1, j - 1) - 2 * grayImg.at<uchar>(i - 1, j) - grayImg.at<uchar>(i - 1, j + 1);

			contour_img.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
		}
	}

	return contour_img;
}

int main() {

	Mat img = imread("C:/Users/User/Desktop/mountain.png");
	Mat gray_img, sepia_img, negative_img, contour_img;
	if (img.empty()) {
		std::cout << "Could not read the image " << std::endl;
		return -1;
	}

#pragma omp parallel sections
	{
#pragma omp section
		gray_img = convert_to_gray(img);
#pragma omp section
		sepia_img = convert_to_sepia(img);
#pragma omp section
		negative_img = convert_to_negative(img);
#pragma omp section
		contour_img = convert_to_contour(img);
	}

	namedWindow("original", WINDOW_NORMAL);
	namedWindow("gray", WINDOW_NORMAL);
	namedWindow("sepia", WINDOW_NORMAL);
	namedWindow("negative", WINDOW_NORMAL);
	namedWindow("contour", WINDOW_NORMAL);

	resizeWindow("original", 800, 600);
	resizeWindow("gray", 800, 600);
	resizeWindow("sepia", 800, 600);
	resizeWindow("negative", 800, 600);
	resizeWindow("contour", 800, 600);

	imshow("original", img);
	imshow("gray", gray_img);
	imshow("sepia", sepia_img);
	imshow("negative", negative_img);
	imshow("contour", contour_img);

	waitKey(0);

	return 0;
}