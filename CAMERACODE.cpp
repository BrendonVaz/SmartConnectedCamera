//------------------------------------------------------------------------------------------------------------------------------
//System Libraries
//------------------------------------------------------------------------------------------------------------------------------
#include <math.h>
#include <errno.h> 
#include <stdio.h>
#include <unistd.h>
#include <memory.h>
#include <string.h> 
#include <stdlib.h>
#include <ctime>
#include <thread>
#include <fstream> 
#include <iostream>
#include <sys/time.h>
#include <sys/types.h> 
//------------------------------------------------------------------------------------------------------------------------------
// TCP/IP Socket Libraries
//------------------------------------------------------------------------------------------------------------------------------
#include <arpa/inet.h> 
#include <sys/socket.h> 
#include <netinet/in.h> 
//------------------------------------------------------------------------------------------------------------------------------
// OpenCV Libraries
//------------------------------------------------------------------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xphoto.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/ximgproc/edge_filter.hpp>
//------------------------------------------------------------------------------------------------------------------------------
using namespace std;
using namespace cv;
using namespace cv::xphoto;
using namespace cv::ximgproc;
//------------------------------------------------------------------------------------------------------------------------------
#define TRUE 	1 
#define FALSE 	0
#define PORT 	8888
//------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------
// Code BEGIN
//------------------------------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------
//~ // Debug Display Thread Function
//------------------------------------------------------------------------------------------------------------------------------
void thread3(const string & Title, Mat showimg, const int t_show){
	//Inputs:	Title: 		Window Display Title
	//			Showimg:	Image to Display
	//			t_show:		Time to Display in seconds
	try {
		if (!showimg.empty()) {
			imshow(Title, showimg); sleep(t_show);
			if (cvGetWindowHandle(Title.c_str())>0){cv::destroyWindow(Title);}
		}
		return;
	}
	catch(int x){destroyAllWindows();return;}
	return;
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Used by Find Circle - Detail Enhance Thread Function
//------------------------------------------------------------------------------------------------------------------------------
void thread4(Mat & srcimg){
	//Inputs:	srcimg: 		address of src/dst img
	Mat tmp = srcimg; detailEnhance(tmp,srcimg,50.0,1.0); return;
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Used by Find Circle - Local contrast enhancement
//------------------------------------------------------------------------------------------------------------------------------
void sauce(Mat & src, Mat & dst, int distance, double sigma){

	CV_Assert(src.type() == CV_8UC1);
	if (!(distance > 0 && sigma > 0)) {
		CV_Error(CV_StsBadArg, "distance and sigma must be greater 0");
	}
	dst = Mat(src.size(), CV_8UC1);
	Mat smoothed;
	int val;
	int a, b;
	int adjuster;
	int half_distance = distance / 2;
	double distance_d = distance;

	GaussianBlur(src, smoothed, cv::Size(0, 0), sigma);

	for (int x = 0;x<src.cols;x++){
		for (int y = 0;y < src.rows;y++) {
			val = src.at<uchar>(y, x);
			adjuster = smoothed.at<uchar>(y, x);
			if ((val - adjuster) > distance_d)adjuster += (val - adjuster)*0.5;
			adjuster = adjuster < half_distance ? half_distance : adjuster;
			b = adjuster + half_distance;
			b = b > 255 ? 255 : b;
			a = b - distance;
			a = a < 0 ? 0 : a;

			if (val >= a && val <= b)
			{
				dst.at<uchar>(y, x) = (int)(((val - a) / distance_d) * 255);
			}
			else if (val < a) {
				dst.at<uchar>(y, x) = 0;
			}
			else if (val > b) {
				dst.at<uchar>(y, x) = 255;
			}
		}
	}
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Used by Find Circle - Tri-Level Thresholding (not used)
//------------------------------------------------------------------------------------------------------------------------------
void tri_threshold(Mat & dst){
	for (int x = 0;x<dst.cols;x++){																			//cycle row
		for (int y = 0;y < dst.rows;y++){																	//cycle col
			dst.at<uchar>(y,x) = (int)((255. / (1.+ exp(-12.*(dst.at<uchar>(y, x) / 255. -0.5))))-1);		//apply sigmoid
			if 		(dst.at<uchar>(y,x) >= 50 && dst.at<uchar>(y,x) <= 150)	{dst.at<uchar>(y, x) = 128;}	//mid range
			else if (dst.at<uchar>(y,x) >= 150)								{dst.at<uchar>(y, x) = 255;}	//high gray
			else if (dst.at<uchar>(y,x) <= 50)								{dst.at<uchar>(y, x) =   0;}	//low gray
		}
	}
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Find Target
//------------------------------------------------------------------------------------------------------------------------------
void FindCircle(double & x, double & y, Mat & capture, int & paws, int & t2show, int & erflag){
	clock_t targettick, targettock;	targettick = clock(); erflag = 0;
	paws = 1; Mat targets = cv::Mat::zeros(cv::Size(1280,480), CV_8UC3);
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Inputs: 	capture		~ address of frame capture	(address ref)
	//~ //			x			~ circle x co-ordinate 		(address ref)
	//~ //			y			~ circle y co-ordinate 		(address ref)
	//~ // 			targets		~ for display and debug 	(address ref)
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Local Variables
	//--------------------------------------------------------------------------------------------------------------------------
	Mat tmp1,tmp2, tmpimg, zumroi, detail, gryimg;																//gray/zoom/detail
	vector<Vec3f> circles1;	vector<Vec3f> circles2;	Point c1;int r1; Point c2;int r2;							//center/radius
	c1.x=0;c1.y=0;c2.x=0;c2.y=0;																				//Initialize Centers
	Point2f	 prs[4]; Point2f	 ROI[4];																		//Region of Interest
	prs[0] = Point2f(320.0-240.0,  0.0);																		//Top left
	prs[1] = Point2f(320.0-240.0,480.0);																		//Bot left
	prs[2] = Point2f(320.0+240.0,  0.0);																		//Top right
	prs[3] = Point2f(320.0+240.0,480.0);																		//Bot right
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Run Algorithm
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Split image into 4 quarters and enhance detail on each part in parallel
	//--------------------------------------------------------------------------------------------------------------------------
	Mat srcimg; while (srcimg.empty()){srcimg = capture.clone();cvtColor(srcimg,tmpimg,COLOR_BGR2GRAY);}		//Get Image Copy
	Mat qrtrtl,qrtrtr,qrtrbl,qrtrbr;																			//declare mat tmps
	qrtrtl = srcimg(cv::Range(  0,240), cv::Range(  0,320));													//img quarters;
	qrtrtr = srcimg(cv::Range(  0,240), cv::Range(320,640));													//img quarters;
	qrtrbl = srcimg(cv::Range(240,480), cv::Range(  0,320));													//img quarters;
	qrtrbr = srcimg(cv::Range(240,480), cv::Range(320,640));													//img quarters;
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //run detail enhance on each quarter in parallel
	//--------------------------------------------------------------------------------------------------------------------------
	thread t4(thread4, ref(qrtrtl));thread t5(thread4, ref(qrtrtr));											//detail on top
	thread t6(thread4, ref(qrtrbl));thread t7(thread4, ref(qrtrbr));											//detail on bot
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //wait for each thread to end and concatenate result
	//--------------------------------------------------------------------------------------------------------------------------
	t4.join();t5.join();t6.join();t7.join();																	//wait for thread
	hconcat(qrtrtl,qrtrtr,tmp1); hconcat(qrtrbl,qrtrbr,tmp2); vconcat(tmp1,tmp2,detail);						//concat result
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // convert to gray scale and find clearance on pcb layer circle using hough
	//--------------------------------------------------------------------------------------------------------------------------
	cvtColor(detail,gryimg,COLOR_BGR2GRAY);	HoughCircles(gryimg,circles1,CV_HOUGH_GRADIENT,1,480,50,25,25,35);
	for (int i=0;i<circles1.size();i++){c1.x = cvRound(circles1[i][0]); c1.y = cvRound(circles1[i][1]); r1 = cvRound(circles1[i][2]);}
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Zoom into ROI target circle if circle found and Find center target using hough
	//--------------------------------------------------------------------------------------------------------------------------
	if (circles1.size() > 0){
		ROI[0] = Point2f(c1.x - 12,c1.y - 12);																		//Top left
		ROI[1] = Point2f(c1.x - 12,c1.y + 12);																		//Bot left
		ROI[2] = Point2f(c1.x + 12,c1.y - 12);																		//Top right
		ROI[3] = Point2f(c1.x + 12,c1.y + 12);																		//Bot right
		Mat zummtx = getPerspectiveTransform(ROI,prs);																//find zoom transform
		warpPerspective(gryimg,zumroi, zummtx, Size(640,480));														//zoom to target
		blur	(zumroi,tmpimg,Size(24,24));																		//blur
		sauce	(tmpimg,zumroi,18,18);																				//contrast
		blur	(zumroi,tmpimg,Size(24,24)); 																		//blur
		//----------------------------------------------------------------------------------------------------------------------
		//~ // Find target circle using hough in region of interest
		//----------------------------------------------------------------------------------------------------------------------	
		HoughCircles(tmpimg,circles2,CV_HOUGH_GRADIENT,0.1,480,50,25,15,90); 
		for (int i=0;i<circles2.size();i++){c2.x = cvRound(circles2[i][0]); c2.y = cvRound(circles2[i][1]); r2 = cvRound(circles2[i][2]);}
		//----------------------------------------------------------------------------------------------------------------------
		//~ // Calculate x,y center solution and set in address x,y
		//----------------------------------------------------------------------------------------------------------------------
		if (c1.x != 0 && c1.y != 0){x=((c1.x-320.0)+(c2.x-320.0)*(12.0/240.0))*(3.0/480.0)*25.4;y=((240.0-c1.y)+(240.0-c2.y)*(12.0/240.0))*(3.0/480.0)*25.4;}
		else {erflag = 1;}
	}
	else {erflag = 1;} targettock = clock(); 
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //End Lock and display Result
	//--------------------------------------------------------------------------------------------------------------------------
	if (t2show > 0){
		cvtColor(tmpimg,zumroi,COLOR_GRAY2BGR);
		for (int i=0;i<circles1.size();i++){circle(srcimg,c1,r1,Scalar(255,0,0), 3,5,0);}
		for (int i=0;i<circles2.size();i++){circle(zumroi,c2,r2,Scalar(255,0,0), 3,5,0);}
		putText(srcimg, to_string(x)+','+to_string(y), Point(25,25), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0));
		hconcat(srcimg,zumroi,targets); thread t3(thread3, "Target Found", targets, 2); t3.detach();
	}
	//--------------------------------------------------------------------------------------------------------------------------
	cout << double(targettock-targettick)/CLOCKS_PER_SEC << "s" << endl;
	cout << "(" << x << "," << y << ")" << endl; paws = 0; return;
	//--------------------------------------------------------------------------------------------------------------------------
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Used by Find Perspective - Get Intersections of Lines
//------------------------------------------------------------------------------------------------------------------------------
void intersection(Point2f &o1, Point2f &p1, Point2f &o2, Point2f &p2, Point2f &xx){
	Point2f x = o2-o1, d1 = p1 - o1, d2 = p2 - o2;
	float cp = d1.x*d2.y - d1.y*d2.x;
	double tt = (x.x*d2.y - x.y*d2.x)/cp;xx = o1+d1*tt; return;
	//~ if (abs(cp) < 0.001) {xx.x = 320;xx.y = 240;return;}
	//~ else                 {double tt = (x.x*d2.y - x.y*d2.x)/cp;xx = o1+d1*tt; return;}
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Find Perspective Transform from Camera to Table
//------------------------------------------------------------------------------------------------------------------------------
void FindSquare(Mat & rmatrix, Mat & rawfram, int & paws, int & t2show, int & erflag){
	paws=1; Mat result = cv::Mat::zeros(cv::Size(1280,480), CV_8UC1);
	clock_t squaretick, squaretock; squaretick = clock(); erflag = 0;
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Inputs: 	capture		~ address of frame capture	(address ref)
	//~ //			rmatrix		~ perspective matrix		(address ref)
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Local Variables
	//--------------------------------------------------------------------------------------------------------------------------
	Point2f	 prs[4];																							//Perspective
	prs[0] = Point2f(320.0-240.0*3.0/3.0,240.0-240.0*3.0/3.0);													//Top left
	prs[1] = Point2f(320.0-240.0*3.0/3.0,240.0+240.0*3.0/3.0);													//Bot left
	prs[2] = Point2f(320.0+240.0*3.0/3.0,240.0-240.0*3.0/3.0);													//Top right
	prs[3] = Point2f(320.0+240.0*3.0/3.0,240.0+240.0*3.0/3.0);													//Bot right
	Point2f	 sqr[4];																							//Corners
	sqr[0] = Point2f(320.0-240.0,  0.0);																		//Top left
	sqr[1] = Point2f(320.0-240.0,480.0);																		//Bot left
	sqr[2] = Point2f(320.0+240.0,  0.0);																		//Top right
	sqr[3] = Point2f(320.0+240.0,480.0);																		//Bot right
	Mat srcimg,dstimg,tmp1,tmp2; while (srcimg.empty()) {srcimg = rawfram.clone();}								//Get Image Copy
	vector<Vec4i> lines; vector<Vec4i> vlines, hlines; vector<Point2f> corners;									//HoughLines, Vertical, Horizontal, Intersections
	vector<Point2f> avgcorn; Mat labels, centers;																//Average Intersections
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Undistort, Grayscale, Threshold, Edge, Find lines
	//--------------------------------------------------------------------------------------------------------------------------
	cvtColor	(srcimg, tmp1, COLOR_BGR2GRAY);
	threshold	(tmp1,tmp2, 0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Canny		(tmp2,tmp1,75,200);
	HoughLinesP	(tmp1,lines,1,CV_PI/180,50,50,10);
	cvtColor	(tmp1, dstimg, COLOR_GRAY2BGR);
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Categorize as Vertical or Horizontal
	//--------------------------------------------------------------------------------------------------------------------------				
	for(size_t i=0;i<lines.size();i++){
		Vec4i l = lines[i]; double theta = (atan2(l[1]-l[3],l[0]-l[2]))*180.0/CV_PI; while(theta<0){theta+=360.0;};
		if ((theta<45.0)||(theta>315.0)||((theta>135.0) && (theta<225.0)))	{hlines.push_back(l);}
		else 																{vlines.push_back(l);}
	}
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Corner Intersection between Vertical and Horizontal Lines
	//--------------------------------------------------------------------------------------------------------------------------				
	for (int i=0;i<hlines.size();i++){
		for (int j=0;j<vlines.size();j++){
			Point2f xx; Vec4i l1 = hlines[i]; Vec4i l2 = vlines[j];
			Point2f o1 = Point2f(l1[0],l1[1]); Point2f p1 = Point2f(l1[2],l1[3]);
			Point2f o2 = Point2f(l2[0],l2[1]); Point2f p2 = Point2f(l2[2],l2[3]);
			intersection(o1, p1, o2, p2, xx);
			corners.push_back(xx);
		}
	}
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Cluster in 4 corners
	//--------------------------------------------------------------------------------------------------------------------------
	if (corners.size()<4) {erflag =1;}//cout << "1111" << endl; return;} //if we do not detect atleast 4 corners,use previous rotation ~ initial set to identity
	kmeans(corners,4,labels,TermCriteria(CV_TERMCRIT_ITER,5,0.5),5,KMEANS_RANDOM_CENTERS,centers);
	avgcorn.push_back(Point2f(centers.at<float>(0,0),centers.at<float>(0,1)));
	avgcorn.push_back(Point2f(centers.at<float>(1,0),centers.at<float>(1,1)));
	avgcorn.push_back(Point2f(centers.at<float>(2,0),centers.at<float>(2,1)));
	avgcorn.push_back(Point2f(centers.at<float>(3,0),centers.at<float>(3,1)));
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Run Sub-pix Corner Detection for corners in image... iff all corners are within image && and no duplicates
	//--------------------------------------------------------------------------------------------------------------------------
	bool run_sub_pix=1;
	for (int i=0;i<4;i++){if (avgcorn[i].x < 0 || avgcorn[i].x > 640 || avgcorn[i].y < 0 || avgcorn[i].y > 480){run_sub_pix=0;break;}}
	for (int i=0;i<4;i++){for (int j=i+1;j<4;j++){double a = avgcorn[i].x-avgcorn[j].x, b = avgcorn[i].y-avgcorn[j].y;if (sqrt(a*a+b*b) < 5) {run_sub_pix=0;break;}}}
	cout << avgcorn << endl;cout << run_sub_pix << endl;
	if (run_sub_pix){cornerSubPix(tmp2,avgcorn,Size(10,10),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,40,0.001));}
	else            {erflag = 1;}// cout << "3333" << endl; return;}
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Arrange Corners in order from top left to bottom right & find perspective transform
	//--------------------------------------------------------------------------------------------------------------------------				
	if (!erflag)
	{
		sort(avgcorn.begin(),avgcorn.end(),[](const Point2f &a, const Point2f &b){return((a.x+a.y)<(b.x+b.y));});
		Point2f tl, br, tr, bl; tl = avgcorn[0]; br = avgcorn[3];
		if (avgcorn[1].x > avgcorn[2].x){tr = avgcorn[1];bl = avgcorn[2];}
		else 							{bl = avgcorn[1];tr = avgcorn[2];}
		sqr[0] = tl;sqr[1] = bl;sqr[2] = tr;sqr[3] = br;
		rmatrix = getPerspectiveTransform(sqr,prs);
	} squaretock = clock();
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // If Requested Draw Results
	//--------------------------------------------------------------------------------------------------------------------------
	if (t2show > 0)
	{
		warpPerspective(srcimg, tmp1, rmatrix, Size(640,480));												//zoom to target
		for(size_t i=0;i<hlines.size();i++){Vec4i l = hlines[i]; line(dstimg,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(0,0,255),3,CV_AA);}
		for(size_t i=0;i<vlines.size();i++){Vec4i l = vlines[i]; line(dstimg,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(255,0,0),3,CV_AA);}
		for(size_t i=0;i<corners.size();i++){ Point p = corners[i]; circle(dstimg,p,2,Scalar(0,255,0),-1,5,0);}
		for(size_t i=0;i<avgcorn.size();i++){ Point p = avgcorn[i]; circle(dstimg,p,10,Scalar(0,255,255),2,5,0);}
		hconcat(dstimg,tmp1,result); thread t3(thread3, "Perspective Found", result, 2); t3.detach();
	}
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //End Lock and display Result
	//--------------------------------------------------------------------------------------------------------------------------
	cout << double(squaretock-squaretick)/CLOCKS_PER_SEC << "s" << endl;
	cout << "(" << rmatrix << ")" << endl;paws = 0; return;
	//--------------------------------------------------------------------------------------------------------------------------
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Used by Find Robot Tooling - Get Angle between two lines, given two points & vertex
//------------------------------------------------------------------------------------------------------------------------------
static double angle( Point pt1, Point pt2, Point pt0 ) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Used by Find Robot Tooling - Get Squares seen in image
//------------------------------------------------------------------------------------------------------------------------------
static void find_robot_tooling_target( const Mat& image, vector<Point> & avgsquare,vector<vector<Point>> & squares,vector<vector<Point>> & contours, vector<Point> & approx) {
    squares.clear(); avgsquare.clear();contours.clear();approx.clear();
	findContours(image, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	//--------------------------------------------------------------------------------------------------------------------------
	// For Found Contours Determine Approximate Contour
	//--------------------------------------------------------------------------------------------------------------------------
	for( size_t i = 0; i < contours.size(); i++ ){
		approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);					
		//----------------------------------------------------------------------------------------------------------------------
		// If Contour Approximation is 4-point, determine if angles between sides = 90
		//----------------------------------------------------------------------------------------------------------------------		
		if( approx.size() == 4 && fabs(contourArea(approx)) > 2000 && isContourConvex(approx)){
			double maxCosine = 0;for(int j=2;j<5;j++){double cosine=fabs(angle(approx[j%4],approx[j-2],approx[j-1]));maxCosine=MAX(maxCosine, cosine);}
			//------------------------------------------------------------------------------------------------------------------
			// If angle is 90 then cosine is 0, then approximate contour is rectangle
			//------------------------------------------------------------------------------------------------------------------
			if( maxCosine < 0.2 ){
				//--------------------------------------------------------------------------------------------------------------
				// For Found Rectangles, Sort Points
				//--------------------------------------------------------------------------------------------------------------
				sort(approx.begin(),approx.end(),[](const Point2f &a,const Point2f &b){return((a.x+a.y)<(b.x+b.y));});
				Point2f tl, br, tr, bl; tl = approx[0]; br = approx[3];
				if (approx[1].x > approx[2].x)	{tr = approx[1];bl = approx[2];}
				else 							{bl = approx[1];tr = approx[2];}
				approx[0]=tl;approx[1]=tr;approx[2]=br;approx[3]=bl;
				//--------------------------------------------------------------------------------------------------------------
				// For Sorted Rectangles, Determine if Sides are Equal
				//--------------------------------------------------------------------------------------------------------------
				double x1 = approx[0].x, y1 = approx[0].y;
				double x2 = approx[1].x, y2 = approx[1].y;
				double x3 = approx[2].x, y3 = approx[2].y;
				double x4 = approx[3].x, y4 = approx[3].y;
				double d1 = sqrt(pow((x2-x1),2)+pow((y2-y1),2));
				double d2 = sqrt(pow((x3-x2),2)+pow((y3-y2),2));
				double d3 = sqrt(pow((x4-x3),2)+pow((y4-y3),2));
				double d4 = sqrt(pow((x4-x1),2)+pow((y4-y1),2));
				if (fabs(1-d1/d2)<0.2 && fabs(1-d1/d3)<0.2 && fabs(1-d1/d4)<0.2){squares.push_back(approx);}
			}
		}
	}
	//--------------------------------------------------------------------------------------------------------------------------
	// For Found & Sorted  Squares Find Average Square
	//--------------------------------------------------------------------------------------------------------------------------
	if (squares.size() > 0){
		Point avg1(0,0), avg2(0,0), avg3(0,0), avg4(0,0);
		for (int i=0;i<squares.size();i++){avg1+=squares[i][0];avg2+=squares[i][1];avg3+=squares[i][2];avg4+=squares[i][3];}
		int len = squares.size(); avg1=avg1/len; avg2=avg2/len; avg3=avg3/len; avg4=avg4/len;
		avgsquare.push_back(avg1); avgsquare.push_back(avg2); avgsquare.push_back(avg3); avgsquare.push_back(avg4);
	}
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Function - Find Perspective Transform from Camera to Robot
//------------------------------------------------------------------------------------------------------------------------------
void FindRobot(double & rx, double & ry, double & rz, Mat & capture, int & paws, int & t2show, int & erflag){
	clock_t robtcptick, robtcptock;	robtcptick = clock(); erflag = 0; paws = 1; 
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Inputs: 	capture		~ address of frame capture	(address ref)
	//~ //			x			~ circle x co-ordinate 		(address ref)
	//~ //			y			~ circle y co-ordinate 		(address ref)
	//~ // 			targets		~ for display and debug 	(address ref)
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Local Variables
	//--------------------------------------------------------------------------------------------------------------------------
	Mat tmp1,tmp2,srcimg;Mat dstimg = cv::Mat::zeros(cv::Size(1280,480), CV_8UC3);rx=0.0;ry=0.0;rz=0.0;
	vector<Point> avgsquare;vector<vector<Point>> squares; vector<vector<Point>> contours; vector<Point> approx;
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Run Algorithm
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Get Image and Find Square
	//--------------------------------------------------------------------------------------------------------------------------
	while (srcimg.empty()){srcimg = capture.clone();cvtColor(srcimg,tmp1,COLOR_BGR2GRAY);}		//Get Image Copy
	threshold	(tmp1,tmp2, 0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	Canny		(tmp2,tmp1,75,200); tmp2 = Mat::zeros(tmp2.size(), CV_8UC3);
	find_robot_tooling_target(tmp1,avgsquare,squares,contours,approx);

	//--------------------------------------------------------------------------------------------------------------------------
	// Find Ratio of Square Size to Actual 1x1 inch target size
	//--------------------------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------------------
	// Find Angle and Center if Square Found
	//--------------------------------------------------------------------------------------------------------------------------

	if (!avgsquare.empty()){

		double x1 = avgsquare[0].x, y1 = avgsquare[0].y;
		double x2 = avgsquare[1].x, y2 = avgsquare[1].y;
		double x3 = avgsquare[2].x, y3 = avgsquare[2].y;
		double x4 = avgsquare[3].x, y4 = avgsquare[3].y;

		double d1 = sqrt(pow((x2-x1),2)+pow((y2-y1),2));
		double d2 = sqrt(pow((x3-x2),2)+pow((y3-y2),2));
		double d3 = sqrt(pow((x4-x3),2)+pow((y4-y3),2));
		double d4 = sqrt(pow((x4-x1),2)+pow((y4-y1),2));

		double rt = 1/(0.25*(d1+d2+d3+d4)*3.0/480.0); 
		
		double a1 = atan2((y1-y2),(x2-x1));
		double a2 = atan2((y4-y3),(x3-x4));
		double a3 = atan2((x3-x2),(y3-y2));
		double a4 = atan2((x4-x1),(y4-y1));

		double x = 0.25*(avgsquare[0].x+avgsquare[1].x+avgsquare[2].x+avgsquare[3].x);
		double y = 0.25*(avgsquare[0].y+avgsquare[1].y+avgsquare[2].y+avgsquare[3].y);

		ry=((240.0-y))*(3.0/480.0)*25.4*rt;	rx=((x-320.0))*(3.0/480.0)*25.4*rt;
		rz = 0.25*(a1+a2+a3+a4)*180.0/CV_PI;robtcptock = clock(); 

		//--------------------------------------------------------------------------------------------------------------------------
		//~ //End Lock and display Result
		//--------------------------------------------------------------------------------------------------------------------------
		if (t2show > 0){
			for (int i=0;i<contours.size();i++){drawContours(srcimg, contours, i, Scalar(25,225,225),3);}
			if (!avgsquare.empty()){
				fillConvexPoly(tmp2, avgsquare, Scalar(255,0,0)); circle(tmp2, Point(x,y), 5, Scalar(0,0,255), 2,8,0);
				putText(tmp2, to_string(rx)+','+to_string(ry)+','+to_string(rz)+','+to_string(rt), Point(25,25), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0));
				hconcat(srcimg,tmp2, dstimg); thread t3(thread3, "Robot Found", dstimg, 2); t3.detach();
			}
		}
	}
	else {erflag = 1;}
	//--------------------------------------------------------------------------------------------------------------------------
	paws = 0; cout << double(robtcptock-robtcptick)/CLOCKS_PER_SEC << "s" << endl;
	cout << "(" << rx << "," << ry << "," << rz << ")" << endl; return;
	//--------------------------------------------------------------------------------------------------------------------------	
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Camera Frame Read Thread
//------------------------------------------------------------------------------------------------------------------------------
void thread1(Mat & capture, Mat & rawfram, clock_t & trigger, Mat & rmatrix, int & init, int & term, int & paws, double & x, double & y, int & t2show, double & rx, double & ry, double & rz){
	//--------------------------------------------------------------------------------------------------------------------------
	//Inputs: 	Cap 		~ video capture object
	//			capture		~ address of frame capture
	//			trigger		~ address of capture time
	//			rmatrix		~ perspective transformation matrix
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Init Start Message
	//--------------------------------------------------------------------------------------------------------------------------
	cout << "-----------------------------------------" << endl;
	cout << "Beginning Camera... " << endl;
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Set up Video Capture
	//--------------------------------------------------------------------------------------------------------------------------
	setUseOptimized(1);
	VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	cap.set(CV_CAP_PROP_FPS,10);
	cap.set(CV_CAP_PROP_BUFFERSIZE,1);
    if (!cap.isOpened()){cerr << "ERROR: Unable to open the camera" << endl; return;}
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Camera Calibration Parameters
	//--------------------------------------------------------------------------------------------------------------------------
    Mat_<float> cammtx(3,3); Mat_<float> dstort(1,5);			//Calibration Matrix; Distortion Matrix
	if (rmatrix.empty()){rmatrix = Mat::eye(3,3,CV_32F);}		//Perspective Matrix; if empty set to identity

	ifstream cammtxfile("cammtx.txt");
	
	for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		cammtxfile >> cammtx[i][j];
	
	ifstream dstortfile("dstort.txt");

	for (int i = 0; i < 5; i++)
		dstortfile >> dstort[0][i];

	cout << "Camera Matrix" << endl;
	cout << cammtx << endl;
	
	cout << "Distortion Co-efficients" << endl;
	cout << dstort << endl;

	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Locals
	//--------------------------------------------------------------------------------------------------------------------------
	Mat tmp1,tmp2, tmp3; int err_count = 0;init =1;
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Init End Message
	//--------------------------------------------------------------------------------------------------------------------------
	cout << "Camera Initialization Complete" << endl;
	cout << "-----------------------------------------" << endl;
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Read Loop
	//--------------------------------------------------------------------------------------------------------------------------
    while(1)
    {	if (term){break;} while(paws){usleep(100);}
		cap >> tmp1; if (tmp1.empty()) {err_count++;cerr<<"Frame Capture Error." << endl; continue;} trigger=clock();
		undistort(tmp1,tmp2,cammtx,dstort); warpPerspective(tmp2,tmp3,rmatrix,tmp1.size());
		tmp2.copyTo(rawfram); tmp3.copyTo(capture);
		if (err_count > 5) break;
	}
	cout <<"Exitting Camera Capture..." << endl;
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Server Socket Thread
//------------------------------------------------------------------------------------------------------------------------------
void thread2(Mat & capture, Mat & rawfram, clock_t & trigger, Mat & rmatrix, int & init, int & term, int & paws, double & x, double & y, int & t2show, double & rx, double & ry, double & rz){
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Server Locals
	//--------------------------------------------------------------------------------------------------------------------------
	int 				opt = TRUE, erflag = 0;
	int 				master_socket,addrlen,new_socket,client_socket[30],max_clients=30,activity,i,valread,sd,max_sd; 
	struct sockaddr_in 	address; 
	char 				buffer[1025];
	char const *		message = "Camera Server\n\r"; 
	fd_set 				readfds; 
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Set up server
	//--------------------------------------------------------------------------------------------------------------------------
	address.sin_family 		= AF_INET; 																			//Protocol
	address.sin_addr.s_addr = INADDR_ANY; 																		//IP Address 
	address.sin_port 		= htons(PORT); 																		//Port Number
	addrlen 				= sizeof(address);																	//Addr Length
	for (i = 0; i < max_clients; i++) {client_socket[i] = 0;} 													//Set Client = 0
	if((master_socket=socket(AF_INET,SOCK_STREAM,0)) == 0)		 						{term=1;cout<<"Error"<<endl;exit(EXIT_FAILURE);}
	if(setsockopt(master_socket,SOL_SOCKET,SO_REUSEADDR,(char *)&opt,sizeof(opt)) < 0)	{term=1;cout<<"Error"<<endl;exit(EXIT_FAILURE);}
	if(bind(master_socket,(struct sockaddr *)&address,addrlen) < 0) 					{term=1;cout<<"Error"<<endl;exit(EXIT_FAILURE);}
	if (listen(master_socket, 3) < 0)													{term=1;cout<<"Error"<<endl;exit(EXIT_FAILURE);} 
	cout << "-----------------------------------------" << endl;
	cout << "Server Set Up Complete... \nListening on port " << PORT << "..." << endl;
	cout << "Waiting for connections ..." << endl; init =1;
	cout << "-----------------------------------------" << endl;
	//--------------------------------------------------------------------------------------------------------------------------
	//~ // Read Loop
	//--------------------------------------------------------------------------------------------------------------------------		
	while(TRUE) {
		if (term){return;} while(paws){usleep(100);}
		//----------------------------------------------------------------------------------------------------------------------
		//~ // Scan Clients Array for Activitiy
		//----------------------------------------------------------------------------------------------------------------------		
		FD_ZERO(&readfds);
		FD_SET(master_socket, &readfds); 
		max_sd = master_socket; 
		for ( i = 0 ; i < max_clients ; i++) {
			sd = client_socket[i]; 
			if(sd > 0)		{FD_SET(sd, &readfds);} 
			if(sd > max_sd)	{max_sd = sd;}	 
		}
		activity = select(max_sd+1,&readfds,NULL,NULL,NULL); 
		if ((activity<0) && (errno!=EINTR)) {cout << "Select Error..." << endl;} 
		//----------------------------------------------------------------------------------------------------------------------
		//~ // If connection Request Accept Client
		//----------------------------------------------------------------------------------------------------------------------					
		if (FD_ISSET(master_socket, &readfds)) { 
			if ((new_socket = accept(master_socket,(struct sockaddr *)&address, (socklen_t*)&addrlen))<0)
				{cout<<"Accept..."<<endl;exit(EXIT_FAILURE);} 
			cout << "New Connection:  " << endl;
			cout << "  Socket Number: " << new_socket << endl;
			cout << "  IP Address:    " << inet_ntoa(address.sin_addr) << endl;
			cout << "  Port Number:   " << ntohs(address.sin_port) << endl;
			if(send(new_socket,message,strlen(message),0) != strlen(message)) {cout << "Sending Message... " << endl;}
			cout << "Confirmation Message Sent Successfully." << endl; 
			for (i = 0; i < max_clients; i++) 
			{ 
				if( client_socket[i] == 0 ) 
				{ 
					client_socket[i] = new_socket; 
					cout << "Connection Added to Client List as ID: " << i << endl; 						
					break; 
				} 
			} 
		}			
		//----------------------------------------------------------------------------------------------------------------------
		//~ // Else: Loop Through Clients
		//----------------------------------------------------------------------------------------------------------------------					
		for (i = 0; i < max_clients; i++) { 
			sd = client_socket[i]; 
			//------------------------------------------------------------------------------------------------------------------
			//~ // If Message
			//------------------------------------------------------------------------------------------------------------------
			if (FD_ISSET(sd, &readfds)) { 
				//--------------------------------------------------------------------------------------------------------------
				//~ // If Failed to Read Message or Force Close
				//--------------------------------------------------------------------------------------------------------------
				if ((valread = read( sd , buffer, 1024)) == 0) { 
					getpeername(sd , (struct sockaddr*)&address , (socklen_t*)&addrlen);
					cout << "Closed Connection: " << endl;
					cout << "  Client Number:   " << i								<< endl;
					cout << "  IP Address:      " << inet_ntoa(address.sin_addr) 	<< endl;
					cout << "  Port Number:     " << ntohs(address.sin_port) 		<< endl;
					close(sd); client_socket[i] = 0; 
				}
				//--------------------------------------------------------------------------------------------------------------
				//~ // Else If Message Successfully Received
				//--------------------------------------------------------------------------------------------------------------
				else { 
					buffer[valread-2] = '\0'; 
					cout << "Client Said: " << endl;
					cout << buffer << endl;
					for (int i = 0; i <= valread-2; i++){cout << "\\x"; cout << hex << int(buffer[i]);cout << " ";};cout << "" << endl;
					for (int i = 0; i <= valread-2; i++){cout << "\\0"; cout << dec << int(buffer[i]);cout << " ";};  cout << "" << endl;
					//----------------------------------------------------------------------------------------------------------
					//~ // If Exit Request, set termination flag
					//----------------------------------------------------------------------------------------------------------
					if (strcmp(buffer,"Exit")==0){term=1;break;}
					//----------------------------------------------------------------------------------------------------------
					//~ // If Ping Request, set termination flag
					//----------------------------------------------------------------------------------------------------------
					else if (strcmp(buffer,"Ping")==0) {
						cout << "Ping Request Received..." << endl;
						send(sd,"Ping Request Received;\n\r",strlen("Ping Request Received;\n\r"),0);
						cout << "Camera Online..." << endl;
						send(sd,"Camera Online;\n\r",strlen("Camera Online;\n\r"),0);
						send(sd,"Good Bye!;\n\r",strlen("Good Bye!;\n\r"),0);
						close(sd);client_socket[i] = 0;
						cout << "Disconnected Client..." << endl;
					}
					//----------------------------------------------------------------------------------------------------------
					//~ // If Calibrate Request
					//----------------------------------------------------------------------------------------------------------
					else if (strcmp(buffer,"Square")==0) {
						cout << "Calibrating Camera..." << endl;
						send(sd,"Finding Square;\n\r",strlen("Finding Square;\n\r"),0);
						FindSquare(rmatrix, rawfram, paws, t2show, erflag);
						cout << "Perspective Calibration Complete..." << endl;
						if (erflag==0) 	{send(sd,"Calibrated Camera;\n\r",strlen("Calibrated Camera;\n\r"),0);}
						else 			{send(sd,"Calibration Error;\n\r",strlen("Calibration Error;\n\r"),0);}
						send(sd,"Good Bye!;\n\r",strlen("Good Bye!;\n\r"),0);
						close(sd);client_socket[i] = 0;
						cout << "Disconnected Client..." << endl;
					}
					//----------------------------------------------------------------------------------------------------------
					//~ // If Target Request
					//----------------------------------------------------------------------------------------------------------
					else if (strcmp(buffer,"Circle")==0) {
						cout << "Finding Target;" << endl;
						send(sd,"Finding Target;\n\r",strlen("Finding Target;\n\r"),0);
						FindCircle(x, y, capture, paws, t2show, erflag);cout << "Sending Result..." << endl;
						if (erflag==0) 	{char res[50]; sprintf(res,"%.3f;%.3f;\n\r\0",x,y); send(sd, res,strlen(res),0);}
						else 			{send(sd,"Error;\n\r",strlen("Error;\n\r"),0);}
						send(sd,"Good Bye!;\n\r",strlen("Good Bye!;\n\r"),0);
						close(sd);client_socket[i] = 0;
						cout << "Disconnected Client..." << endl;
					}
					//----------------------------------------------------------------------------------------------------------
					//~ // If Robot TCP Request
					//----------------------------------------------------------------------------------------------------------					
					else if (strcmp(buffer,"RobTCP")==0) {
						cout << "Finding Robot Tool Target..." << endl;
						send(sd,"Finding Robot Tool Target;\n\r",strlen("Finding Robot Tool Target;\n\r"),0);
						FindRobot(rx, ry, rz, capture, paws, t2show, erflag);cout << "Sending Result..." << endl;
						if (erflag==0) 	{char res[50]; sprintf(res,"%.3f;%.3f;%.3f;\n\r\0",rx,ry,rz); send(sd, res,strlen(res),0);}
						else 			{send(sd,"Error;\n\r",strlen("Error;\n\r"),0);}
						send(sd,"Good Bye!;\n\r",strlen("Good Bye!;\n\r"),0);
						close(sd);client_socket[i] = 0;
						cout << "Disconnected Client..." << endl;
					}
					//----------------------------------------------------------------------------------------------------------
					//~ // If Invalid Request
					//----------------------------------------------------------------------------------------------------------
					else {
						cout << "Invalid Command Received, Removing Client..." << endl;
						send(sd,"Invalid Command!...",strlen("Invalid Command!..."),0);
						send(sd,"Good Bye!...\n\r",strlen("Good Bye!...\n\r"),0);
						close(sd);client_socket[i] = 0;
						cout << "Disconnected Client..." << endl;
					}
				}
			} 
		} 
	} 
	cout << "Exitting Server... " << endl;
	return; 
}

//------------------------------------------------------------------------------------------------------------------------------
//~ // Main Thread
//------------------------------------------------------------------------------------------------------------------------------
int main (int argc, char *argv[]){
	//--------------------------------------------------------------------------------------------------------------------------
	//Handle Command Line Debug Show Flag
	//--------------------------------------------------------------------------------------------------------------------------
	int showtime = 0; int dbugtime = 0; string fn = argv[0];
	if (argc > 3)	{cout << "Only Two Arguments Accepted: \nCapture Preview Time in seconds\nResults Display Time in seconds (1-5 s)" << endl;return 0;}
	if (argc > 2)	{dbugtime = atoi(argv[2]);}		//-> preview for requested time
	if (argc > 1)	{showtime = atoi(argv[1]);}		//-> preview for requested time
	if (dbugtime>5 || showtime==-1){dbugtime=5;}	//-> if continous playback, show answer for 5 sec
	if (argc >0){
		cout << "-----------------------------------------" << endl;
		cout << "Filename:\t"   << fn 						<< endl;
		cout << "Input Cnt:\t"  << argc 					<< endl;
		cout << "Preview(s):\t" << showtime 				<< endl;
		cout << "Results(s):\t" << dbugtime 				<< endl;
		cout << "-----------------------------------------" << endl;
	}
	//--------------------------------------------------------------------------------------------------------------------------
	// Thread Control Variables
	//--------------------------------------------------------------------------------------------------------------------------
	int init = 0;						//Initialization flag: 	passed to frame capture thread and server thread
	int paws = 0;						//Pause Request flag:	Passed to frame capture thread and server thread
	int term = 0;						//Termination flag:		Passed to frame capture thread and server thread
	//--------------------------------------------------------------------------------------------------------------------------
	// Variables used by multiple functions
	//--------------------------------------------------------------------------------------------------------------------------
	Mat capture; 						//Frame Capture Matrix:	Passed to frame capture thread and server thread -> Find Feature
	double x,y;							//Circle Center Result:	Passed to server thread 						 -> Find Circle
	double rx,ry,rz;					//Robot  Center Result:	Passed to server thread 						 -> Find Robot
	Mat rawfram;						//Raw Capture Matrix:	Passed to frame capture thread and server thread -> Find Square
	Mat rmatrix;						//Perspective Result: 	Passed to frame capture thread and server thread -> Find Square	
	clock_t trigger;					//Frame Capture Timer:	Passed to frame capture thread and server thread
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Start Camera thread, wait while initializing
	//--------------------------------------------------------------------------------------------------------------------------
	thread t1(thread1, ref(capture), ref(rawfram), ref(trigger), ref(rmatrix), ref(init),ref(term), ref(paws), ref(x),ref(y),ref(dbugtime),ref(rx), ref(ry), ref(rz)); while (!init){continue;} 
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Start Server thread, wait while initializing
	//--------------------------------------------------------------------------------------------------------------------------
	thread t2(thread2, ref(capture), ref(rawfram), ref(trigger), ref(rmatrix), ref(init),ref(term), ref(paws), ref(x),ref(y),ref(dbugtime),ref(rx), ref(ry), ref(rz)); while (!init){continue;} 
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //If Show Live... for specified amount of time
	//--------------------------------------------------------------------------------------------------------------------------
	clock_t showclock = clock(); sleep(2.5);
	if (showtime != 0)
	{
		while (1) {
			while (paws){usleep(100);}
			imshow("Capture: Press Key to Kill Preview", capture);
			int key=cv::waitKey(5);key=(key==255)?-1:key;
			if(key>=0) {if (cvGetWindowHandle("Capture: Press Key to Kill Preview")>0){destroyWindow("Capture: Press Key to Kill Preview");} break;}
			if ((showtime!=-1) && (double(clock()-showclock)/CLOCKS_PER_SEC > showtime)) {	//if user specified debug show time expired
				if (cvGetWindowHandle("Capture: Press Key to Kill Preview")>0){destroyWindow("Capture: Press Key to Kill Preview");} break;}
			if (term){break;}
		}
	}
	else{cout << "Display Window Muted, Camera Running..." << endl;}
	//--------------------------------------------------------------------------------------------------------------------------
	//~ //Kill and Exit...
	//--------------------------------------------------------------------------------------------------------------------------
	t1.join(); t2.join();
	cout << "Exitted Camera Thread" 			<< endl;
	cout << "Exitted Server Thread" 			<< endl;
	cout << "Exitted Main with Error Code: 0"	<< endl;
	return 0;
}
//------------------------------------------------------------------------------------------------------------------------------
