#include"morphing.h"

using namespace std;



int counter = 0;
int frame_count = 0;
IplImage *leftImage, *rightImage;
IplImage *leftImageTmp, *rightImageTmp;
int height = 0;
int width = 0;
CvScalar Color = CV_RGB(0, 255, 0);//框框顏色
int Thickness = 2;//框框粗細
int Shift = 0;//框框大小(0為正常)
int key;//按鍵碼

double parameter_a = 0.0;
double parameter_b = 0.0;
double parameter_p = 0.0;

string first_image_name;
string second_image_name;

std::vector<struct LinePair> pairs;
LinePair curLinePair;


void Line::PQtoMLD()
{
	//CvPoint2D32f tmpP=cvPointTo32f(P);
	//CvPoint2D32f tmpQ=cvPointTo32f(Q);
	M.x = (P.x + Q.x) / 2;
	M.y = (P.y + Q.y) / 2;

	float tmpx = Q.x - P.x;
	float tmpy = Q.y - P.y;

	len = sqrt(tmpx*tmpx + tmpy*tmpy);
	degree = atan2(tmpy, tmpx);
	return;
}
void Line::MLDtoPQ()
{
	float tmpx = 0.5*len*cos(degree);
	float tmpy = 0.5*len*sin(degree);

	CvPoint2D32f tmpP;
	CvPoint2D32f tmpQ;
	tmpP.x = M.x - tmpx;
	tmpP.y = M.y - tmpy;
	tmpQ.x = M.x + tmpx;
	tmpQ.y = M.y + tmpy;

	P = tmpP;
	Q = tmpQ;
	return;
}
void Line::show()
{
	printf("P(%lf,%lf) Q(%lf,%lf) M(%lf,%lf)\n \tlen=%lf degree=%f\n", P.x, P.y, Q.x, Q.y, M.x, M.y, len, degree);
	return;
}
double Line::Getu(CvPoint2D32f X) {
	/*calculate u*/
	double X_P_x = X.x - P.x;
	double X_P_y = X.y - P.y;
	double Q_P_x = Q.x - P.x;
	double Q_P_y = Q.y - P.y;
	double u = ((X_P_x * Q_P_x) + (X_P_y * Q_P_y)) / (len * len);
	return u;
}
double Line::Getv(CvPoint2D32f X) {
	double X_P_x = X.x - P.x;
	double X_P_y = X.y - P.y;
	double Q_P_x = Q.x - P.x;
	double Q_P_y = Q.y - P.y;
	double Perp_Q_P_x = Q_P_y;
	double Perp_Q_P_y = -Q_P_x;
	double v = ((X_P_x * Perp_Q_P_x) + (X_P_y * Perp_Q_P_y)) / len;
	return v;
}
CvPoint2D32f Line::Get_Point(double u, double v) {
	double Q_P_x = Q.x - P.x;
	double Q_P_y = Q.y - P.y;
	double Perp_Q_P_x = Q_P_y;
	double Perp_Q_P_y = -Q_P_x;
	double Point_x = P.x + u * (Q.x - P.x) + ((v * Perp_Q_P_x) / len);
	double Point_y = P.y + u * (Q.y - P.y) + ((v * Perp_Q_P_y) / len);
	CvPoint2D32f X;
	X.x = Point_x;
	X.y = Point_y;
	return X;
}
double Line::Get_Weight(CvPoint2D32f X) {
	double a = parameter_a;
	double b = parameter_b;
	double p = parameter_p;
	double d = 0.0;

	double u = Getu(X);
	if (u > 1.0)
		d = sqrt((X.x - Q.x) * (X.x - Q.x) + (X.y - Q.y) * (X.y - Q.y));
	else if (u < 0)
		d = sqrt((X.x - P.x) * (X.x - P.x) + (X.y - P.y) * (X.y - P.y));
	else
		d = abs(Getv(X));


	double weight = pow(pow(len, p) / (a + d), b);
	return weight;
}
//======================================
void LinePair::genWarpLine()
{
	while (leftLine.degree - rightLine.degree>3.14159265)
		rightLine.degree = rightLine.degree + 3.14159265;
	while (rightLine.degree - leftLine.degree>3.14159265)
		leftLine.degree = leftLine.degree + 3.14159265;
	for (int i = 0; i<frame_count; i++)
	{
		double ratio = (double)(i + 1) / (frame_count + 1);
		Line curLine;

		curLine.M.x = (1 - ratio)*leftLine.M.x + ratio*rightLine.M.x;
		curLine.M.y = (1 - ratio)*leftLine.M.y + ratio*rightLine.M.y;
		curLine.len = (1 - ratio)*leftLine.len + ratio*rightLine.len;
		curLine.degree = (1 - ratio)*leftLine.degree + ratio*rightLine.degree;

		curLine.MLDtoPQ();
		warpLine.push_back(curLine);
	}
	return;
}
void LinePair::showWarpLine()
{
	int size = warpLine.size();
	for (int i = 0; i<size; i++)
	{
		printf("warpLine[%d]:", i);
		warpLine[i].show();
		cvLine(leftImage, cvPointFrom32f(warpLine[i].P), cvPointFrom32f(warpLine[i].Q), Color, Thickness, CV_AA, Shift);
		cvLine(rightImage, cvPointFrom32f(warpLine[i].P), cvPointFrom32f(warpLine[i].Q), Color, Thickness, CV_AA, Shift);
	}
	leftImageTmp = cvCloneImage(leftImage);
	rightImageTmp = cvCloneImage(rightImage);
	cvShowImage("left", leftImage);
	cvShowImage("right", rightImage);
	return;
}






struct Image{
    int frame_index;
    IplImage *image;
    IplImage *new_image;
    IplImage *test_image;
    
    Image(int i);
    void LoadImage(string image_name);
    CvScalar bilinear(IplImage *image , double X  , double Y );
    void Warp();
};
Image::Image(int i){
	frame_index=i;
    CvSize ImageSize = cvSize(width,height);
    new_image = cvCreateImage(ImageSize,IPL_DEPTH_8U,3);
    test_image = cvCreateImage(ImageSize,IPL_DEPTH_8U,3);
}
CvScalar Image::bilinear(IplImage *image , double X  , double Y ){
     int x_floor = (int)X ;
     int y_floor = (int)Y ;
     int x_ceil = x_floor + 1 ;
     int y_ceil = y_floor + 1 ;
     double a = X - x_floor ;
     double b = Y - y_floor ;
     if(x_ceil >= width-1) 
         x_ceil = width-1 ;
     if(y_ceil >= height-1) 
         y_ceil = height-1 ;
     CvScalar output_scalar ;
     CvScalar leftdown = cvGet2D(image,y_floor,x_floor);
     CvScalar lefttop = cvGet2D(image,y_ceil,x_floor);
     CvScalar rightdown = cvGet2D(image,y_floor,x_ceil);
     CvScalar righttop = cvGet2D(image,y_ceil,x_ceil);
     for(int i = 0 ; i < 4 ; i ++){
             output_scalar.val[i] = (1-a)*(1-b)*leftdown.val[i] + a*(1-b)*rightdown.val[i] + a*b*righttop.val[i] + (1-a)*b*lefttop.val[i];
     }
     return output_scalar ;
}
void Image::Warp(){
	double ratio=(double)(frame_index+1)/(frame_count+1);
	
	IplImage *ori_leftImage,*ori_rightImage;
	ori_leftImage=cvLoadImage(first_image_name.c_str());
  	ori_rightImage=cvLoadImage(second_image_name.c_str());

	//cvShowImage("win_name", ori_rightImage);
	//cvWaitKey(0);

    for(int x = 0 ; x < width ; x++){
        for(int y = 0 ; y < height ; y++){
            CvPoint2D32f dst_point ;
            dst_point.x = x ; 
            dst_point.y = y;
            double leftXSum_x = 0.0;
            double leftXSum_y = 0.0;
            double leftWeightSum = 0.0;
            double rightXSum_x = 0.0;
            double rightXSum_y = 0.0;
            double rightWeightSum = 0.0;
            for(int i = 0 ; i < pairs.size() ; i++){
				//左圖為來源 
				Line src_line = pairs[i].leftLine;	
                Line dst_line = pairs[i].warpLine[frame_index];
                
                double new_u = dst_line.Getu(dst_point);
                double new_v = dst_line.Getv(dst_point);
                
                CvPoint2D32f src_point = src_line.Get_Point(new_u , new_v);
                double src_weight = dst_line.Get_Weight(dst_point);
                leftXSum_x = leftXSum_x + (double)src_point.x * src_weight ;
                leftXSum_y = leftXSum_y + (double)src_point.y * src_weight ;
                leftWeightSum = leftWeightSum + src_weight ;
                
                //右圖為來源 
                src_line = pairs[i].rightLine;	
                
                new_u = dst_line.Getu(dst_point);
                new_v = dst_line.Getv(dst_point);
                
                src_point = src_line.Get_Point(new_u , new_v);
                src_weight = dst_line.Get_Weight(dst_point);
                rightXSum_x = rightXSum_x + (double)src_point.x * src_weight ;
                rightXSum_y = rightXSum_y + (double)src_point.y * src_weight ;
                rightWeightSum = rightWeightSum + src_weight ;
            }
            double left_src_x = leftXSum_x / leftWeightSum;
            double left_src_y = leftXSum_y / leftWeightSum;
            double right_src_x = rightXSum_x / rightWeightSum;
            double right_src_y = rightXSum_y / rightWeightSum;
            
            /*邊界*/
            if(left_src_x<0)
            	left_src_x=0;
            if(left_src_y<0)
            	left_src_y=0;
			if(left_src_x>=width)
            	left_src_x=width-1;
            if(left_src_y>=height)
            	left_src_y=height-1;
            if(right_src_x<0)
            	right_src_x=0;
            if(right_src_y<0)
            	right_src_y=0;
			if(right_src_x>=width)
            	right_src_x=width-1;
            if(right_src_y>=height)
            	right_src_y=height-1;
            
            
			//CvScalar left_scalar=cvGet2D(ori_leftImage,left_src_y,left_src_x);
			//CvScalar right_scalar=cvGet2D(ori_rightImage,right_src_y,right_src_x);
			CvScalar left_scalar = bilinear(ori_leftImage,left_src_x,left_src_y);
			CvScalar right_scalar=bilinear(ori_rightImage,right_src_x,right_src_y);
			CvScalar new_scalar;
			new_scalar.val[0]=(1-ratio)*left_scalar.val[0] + ratio*right_scalar.val[0];
			new_scalar.val[1]=(1-ratio)*left_scalar.val[1] + ratio*right_scalar.val[1];
			new_scalar.val[2]=(1-ratio)*left_scalar.val[2] + ratio*right_scalar.val[2];
			new_scalar.val[3]=(1-ratio)*left_scalar.val[3] + ratio*right_scalar.val[3];
            cvSet2D(new_image,y,x,new_scalar);
            cvSet2D(test_image,y,x,left_scalar);//test
            
            /*
            CvScalar Color=CV_RGB(0,255,0);//框框顏色
            int Thickness=2;//框框粗細
            int Shift=0;//框框大小(0為正常)
            cvLine(test_image,pairs[0].warpLine[frame_index].P,pairs[0].warpLine[frame_index].Q,Color,Thickness,CV_AA,Shift);
		    */
        }
        
    }
    //char win_name[16];
    //char img_name[50];
	//char test_name[16];
    //sprintf(win_name,"frame[%d]",frame_index);
    //sprintf(img_name,"%s_%d.jpg",new_image_name.c_str(),frame_index);
    //sprintf(test_name,"test[%d]",frame_index);

	//cvShowImage(win_name,new_image); 
    //cvSaveImage(img_name,new_image);
	//cvShowImage(test_name,test_image);

	char win_name[] = "MORPHING";
	cvShowImage(win_name,new_image); 

	cvWaitKey(10);

    return ;
}
//======================================

void runWarp()
{
	for(int i=0;i<frame_count;i++)
	{
		Image curImage=Image(i);
		printf("warping %d...\n",i);
		curImage.Warp();
	}	
}

void show_pairs()
{
	int len = pairs.size();
	printf("pairs size=%d\n", len);
	for (int i = 0; i<len; i++)
	{
		printf("leftLine:");
		pairs[i].leftLine.show();
		printf("rightLine:");
		pairs[i].rightLine.show();
		printf("\n");
	}
}



