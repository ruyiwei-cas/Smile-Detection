#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include<fstream>
#include<vector>
#include<string>

using namespace cv;
using namespace std;

#define ATTRIBUTE_FEATURE_DIM 512

#define FACE_ID_THRESHOLD 0.1
#define MAX_FACE_ID 20
#define FACE_TEMPLATE_MAX_NUM  MAX_FACE_ID
#define MAX_FACE_NUMBER   1600


#define size_smallface  64
#define size_bigface    128

#define ATTRIBUTE_FEATURE_DIM 512


enum EnumViewAngle{
	// multi-profile detection
	VIEW_ANGLE_0 = 0x00000001,
	VIEW_ANGLE_45 = 0x00000002,
	VIEW_ANGLE_90 = 0x00000004,
	VIEW_ANGLE_135 = 0x00000008,
	VIEW_ANGLE_180 = 0x00000010,

	// multi-roll detection
	VIEW_ROLL_30 = 0x00000020,
	VIEW_ROLL_30N = 0x00000040,
	VIEW_ROLL_60 = 0x00000080,
	VIEW_ROLL_60N = 0x00000100,

	VIEW_ANGLE_FRONTAL = 0x00000004,
	VIEW_ANGLE_HALF_MULTI = 0x0000000E,
	VIEW_ANGLE_MULTI = 0x0000001F,

	VIEW_ANGLE_FRONTALROLL = 0x000001E4,
	VIEW_ANGLE_HALF_MULTI_FRONTALROLL = VIEW_ANGLE_HALF_MULTI | VIEW_ANGLE_FRONTALROLL,
	VIEW_ANGLE_MULTI_FRONTALROLL = VIEW_ANGLE_MULTI | VIEW_ANGLE_FRONTALROLL,

	VIEW_ANGLE_OMNI = 0xFFFFFFFF,
};

class CvRectItem
{
public:
	CvRectItem() { prob = 0; rc = cvRect(0, 0, 0, 0); vid = VIEW_ANGLE_90;  angle = 0; neighbors = 0; fid = -1; }

	CvRect             rc;		// region
	float              prob;	// probability
	int          	   vid;		// view-id
	int                fid;     // face-id
	int				   angle;	// roll-angle
	int                neighbors;
	int                reserved[8]; // reserved[0] which stage the rectange in
};

typedef struct Face_Attribute
{
	CvRect FaceRegion;
	int FaceView;

	int FaceID;
	float Prob_FaceID;

	int	   Blink;
	float  Prob_Blink;
	int	   Age;
	float  Prob_Age;
	int	   Smile;
	float  Prob_Smile;
	int	   Gender;
	float  Prob_Gender;
	double Attribute_Feature[ATTRIBUTE_FEATURE_DIM];
} Face_Attribute;

void FaceID_PostProcessing2(Face_Attribute *FaceRecognitionResult, int *Face_Valid_Flag, int nface_num);
bool FaceRecognitionApplication(IplImage* color_image, int nFace_Num, CvRectItem* rects);
int FaceDetectionApplication(IplImage* color_image, CvRectItem* rects, int MAX_face_numBER, bool bRotateTry);
void InitFaceDetector();