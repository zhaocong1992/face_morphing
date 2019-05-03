#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include<string.h>
#include<vector>
#include<io.h>
#include"morphing.h"


using namespace cv;
using namespace std;
using namespace dlib;

extern int frame_count;
extern IplImage *leftImage, *rightImage;
extern IplImage *leftImageTmp, *rightImageTmp;
extern int height;
extern int width;

extern string first_image_name;
extern string second_image_name;
extern string new_image_name;

extern double parameter_a;
extern double parameter_b;
extern double parameter_p;

extern std::vector<struct LinePair> pairs;
extern LinePair curLinePair;

static void getlinesFromShapes(std::vector<full_object_detection> &shapes, std::vector<LinePair> &pairs)
{
	std::vector<std::vector<float>> lines_select = {
		{ 0,2 },{ 3,4 },{ 5,6 },{ 7,8 },{ 8,9 },{ 10,11 },{ 12,13 },{ 14,16 },
		{ 17,19 },{ 19,21 },{ 22,24 },{ 24,26 },{ 36,39 },{ 37,41 },{ 38,40 },
		{ 42,45 },{ 43,47 },{ 44,46 },{ 27,30 },{ 31,33 },{ 48,51 },{ 51,54 },
		{ 48,57 },{ 56,54 } };


	for (int i = 0; i < lines_select.size(); i++)
	{
		int satrt_ind = lines_select[i][0];
		int end_ind = lines_select[i][1];

		curLinePair.warpLine.clear();

		curLinePair.leftLine.P.x = (float)shapes[0].part(satrt_ind).x();
		curLinePair.leftLine.P.y = (float)shapes[0].part(satrt_ind).y();
		curLinePair.leftLine.Q.x = (float)shapes[0].part(end_ind).x();
		curLinePair.leftLine.Q.y = (float)shapes[0].part(end_ind).y();
		curLinePair.leftLine.PQtoMLD();

		curLinePair.rightLine.P.x = (float)shapes[1].part(satrt_ind).x();
		curLinePair.rightLine.P.y = (float)shapes[1].part(satrt_ind).y();
		curLinePair.rightLine.Q.x = (float)shapes[1].part(end_ind).x();
		curLinePair.rightLine.Q.y = (float)shapes[1].part(end_ind).y();
		curLinePair.rightLine.PQtoMLD();

		curLinePair.genWarpLine();

		pairs.push_back(curLinePair);

	}
	return;
}

void getFiles(string path, std::vector<string>& files)
{
	//文件句柄  
	intptr_t   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (!(fileinfo.attrib &  _A_SUBDIR))
			{
				files.push_back(fileinfo.name);
			}

		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


int main()
{

	string src_path = "..\\images";

	std::vector<string> files_names;
	getFiles(src_path, files_names);


	//set paras
	parameter_a = 1;
	parameter_b = 2;
	parameter_p = 3;
	frame_count = 30;

	if (files_names.size() < 2)
	{
		cout << "No enough pics for morphing!" << endl;
		return 0;
	}

	for (int i = 0; i < files_names.size() -1; i++)
	{
		cout << "Morphing from " << files_names[i] << " to " << files_names[i+1] << endl;

		//init morphing paras
		first_image_name = src_path + "\\" + files_names[i];
		second_image_name = src_path + "\\" + files_names[i+1];

		leftImage = cvLoadImage(first_image_name.c_str());
		rightImage = cvLoadImage(second_image_name.c_str());
		height = leftImage->height;
		width = leftImage->width;
		leftImageTmp = cvLoadImage(first_image_name.c_str());
		rightImageTmp = cvLoadImage(second_image_name.c_str());


		Mat src_img = imread(first_image_name);
		Mat dst_img = imread(second_image_name);

		// Load face detection and pose estimation models.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		// Turn OpenCV's Mat into something dlib can deal with.
		cv_image<bgr_pixel> src_cimg(src_img);
		cv_image<bgr_pixel> dst_cimg(dst_img);

		// Detect faces 
		std::vector<dlib::rectangle> src_faces = detector(src_cimg);
		std::vector<dlib::rectangle> dst_faces = detector(dst_cimg);
		// Find the pose of each face.
		full_object_detection src_shape, dst_shape;
		src_shape = pose_model(src_cimg, src_faces[0]);
		dst_shape = pose_model(dst_cimg, dst_faces[0]);

		std::vector<full_object_detection> shapes;
		shapes.push_back(src_shape);
		shapes.push_back(dst_shape);

		getlinesFromShapes(shapes, pairs);

		cvNamedWindow("src_img", 0);
		imshow("src_img", src_img);
		cvWaitKey(10);

		for (int i = 0; i < 68; i++) {
			circle(src_img, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 1, cv::Scalar(0, 0, 255), -1);
			shapes[0].part(i).x();//68个
		}
		imshow("src_img", src_img);
		cvWaitKey(10);

		cvNamedWindow("dst_img", 0);
		imshow("dst_img", dst_img);
		cvWaitKey(10);

		for (int i = 0; i < 68; i++) {
			circle(dst_img, cvPoint(shapes[1].part(i).x(), shapes[1].part(i).y()), 1, cv::Scalar(0, 0, 255), -1);
			shapes[0].part(i).x();//68个
		}
		imshow("dst_img", dst_img);
		cvWaitKey(100);

		runWarp();

	}

	return 0;

}
