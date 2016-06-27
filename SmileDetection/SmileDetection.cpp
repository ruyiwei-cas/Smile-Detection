#include"SmileDetection.h"
#include"basetypes.hpp"
#include"cxlibface.hpp"

#define MAX_FACE_NUMBER   1600

int Face_Valid_Flag[MAX_FACE_NUMBER];

Face_Attribute FaceRecognitionResult[MAX_FACE_NUMBER];
CxlibLandmarkDetector landmarkDetector(LDM_6PT);
CxlibAlignFace cutFace(size_smallface, size_bigface);
CxlibSmileDetector  smileDetector(size_smallface);
CxlibFaceRecognizer faceRecognizer(size_bigface);
CxlibFaceDetector detector;
////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face post processing
///  written by Ren Haibing 
/// Acknowledge		: None
///
/// Function name	: FaceID_PostProcessing
/// Description	    : reject the imposters
///
/// Argument		:	FaceRecognitionResult -- recognition result
/// Argument		:	Face_Valid_Flag -- valid face flag
/// Argument		:	nface_num -- detected face number
///
/// Return type		: 
///
/// Create Time		: 2014-11-4  13:01
///
///
/// Side Effect		: 
///                   
///////////////////////////////////////////////////////////////////////////////////////////////

void FaceID_PostProcessing2(Face_Attribute *FaceRecognitionResult, int *Face_Valid_Flag, int nface_num)
{
	int FaceID_Array[MAX_FACE_ID];
	float FaceID_MaxProb[MAX_FACE_ID];
	int i;

	for (i = 0; i<MAX_FACE_ID; i++)
	{
		FaceID_Array[i] = 0;
		FaceID_MaxProb[i] = 0.0f;
	}
	int nFaceID;

	//2. check if 2 face images have the same ID
	for (i = 0; i<nface_num; i++)
	{
		if (Face_Valid_Flag[i] != 1)
			continue;
		nFaceID = FaceRecognitionResult[i].FaceID;
		FaceID_Array[nFaceID]++;
		if (FaceID_MaxProb[nFaceID] < FaceRecognitionResult[i].Prob_FaceID)
			FaceID_MaxProb[nFaceID] = FaceRecognitionResult[i].Prob_FaceID;
	}

	for (i = 0; i<nface_num; i++)
	{
		if (Face_Valid_Flag[i] != 1)
			continue;
		nFaceID = FaceRecognitionResult[i].FaceID;
		if (nFaceID<0) continue;
		if (FaceID_Array[nFaceID]>1) // more than 1 images have the same ID
		{
			if (FaceRecognitionResult[i].Prob_FaceID < FaceID_MaxProb[nFaceID] - 0.001) // only remove the one with small probility
				FaceRecognitionResult[i].FaceID = -1;  //face name is set as "N/A"		
		}
	}

	//1. remove the result when confidence < threshold
	for (i = 0; i<nface_num; i++)
	{
		if (Face_Valid_Flag[i] != 1)
			continue;
		if (FaceRecognitionResult[i].Prob_FaceID<FACE_ID_THRESHOLD)
			FaceRecognitionResult[i].FaceID = -1;  //face name is set as "N/A"
	}

}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Recognition Utility
///  written by Ren Haibing 
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: FaceRecognitionApplication
/// Description	    : face identification and face attribute recognition 
///
/// Argument		:	color_image -- source image
/// Argument		:	nFace_Num -- detected face number
/// Argument		:	rects -- detected face region
///
/// Return type		: 
///
/// Create Time		: 2014-11-5  10:56
///
///
/// Side Effect		: int Face_Valid_Flag[MAX_face_numBER] -- face valid flag array
///                   Face_Attribute FaceRecognitionResult[MAX_face_numBER] -- final recogntion result 
///////////////////////////////////////////////////////////////////////////////////////////////

bool FaceRecognitionApplication(IplImage* color_image, int nFace_Num, CvRectItem* rects)
{
	bool   DoBlink = true, DoSmile = true, DoGender = true, DoAge = true;
	float  smile_threshold, blink_threshold, gender_threshold;
	int    bBlink = 0, bSmile = 0, bGender = 0;  //+1, -1, otherwise 0: no process 
	int    nAgeID = 0;
	float  probBlink = 0, probSmile = 0, probGender = 0, probAge[4];
	bool smileFlag = false;
	// config landmark detector ------------------------------------


	bool  bLandmark = false;
	CvPoint2D32f   landmark6[6 + 1]; // consider both 6-pt and 7-pt

	float probFaceID;
	int nFaceSetID;

	// blink/smile/gender/age/face recognize section
	for (int i = 0; i< nFace_Num; i++)
	{
		Face_Valid_Flag[i] = 0;
		bSmile = bBlink = bGender = -1;
		probSmile = 0, probBlink = 0, probGender = 0;

		// get face rect and id from face tracker
		CvRect rect = rects[i].rc;
		int    face_trackid = rects[i].fid;
		float  like = rects[i].prob;
		int    angle = rects[i].angle;

		FaceRecognitionResult[i].FaceRegion = rect;
		FaceRecognitionResult[i].FaceView = 0;//frontal view

		// filter out outer faces
		if (rect.x + rect.width  > color_image->width || rect.x < 0) continue;
		if (rect.y + rect.height > color_image->height || rect.y < 0) continue;
		if (rect.width<color_image->width * 0.03) continue;

		// Landmark detection -----------------------------------------------------
		bLandmark = landmarkDetector.detect(color_image, &rect, landmark6, NULL, angle); //for imagelist input
		if (bLandmark == false) continue;
		cutFace.init(color_image, rect, landmark6);

		Face_Valid_Flag[i] = 1;

		
		// detect smile -----------------------------------------------------------
		bSmile = 0;
		probSmile = 0;
		if (DoSmile)
		{
			smile_threshold = smileDetector.getDefThreshold(); //0.42;  
			int ret = smileDetector.predict(&cutFace, &probSmile);

			if (probSmile > smile_threshold)
			{
				bSmile = 1;
				smileFlag = true;
			}//smile
			else
				bSmile = 0; //not smile
			FaceRecognitionResult[i].Smile = bSmile;
			FaceRecognitionResult[i].Prob_Smile = probSmile;
		}

		
	}//for( int i=0; i< nFace_Num; i++ )
	return smileFlag;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: FaceDetectionApplication
/// Description	    : face detection and rotate the image with face detection result(in the scenio of no exif file)
///
/// Argument		:	color_image -- source color image
/// Argument		:	gray_image -- source gray image
/// Argument		:	rects -- detected face region
/// Argument		:	MAX_face_numBER -- maximal face number
/// Argument		:	imgExif -- image exif information
///
/// Return type		:  int -- detected face number
///
/// Create Time		: 2014-10-28  16:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
#define LARGE_IMAGE_SIZE  20000
#define STANDARD_IMAGE_WIDTH_LONG  2000
#define STANDARD_IMAGE_WIDTH_SMALL  1200
int FaceDetectionApplication(IplImage* color_image, CvRectItem* rects, int MAX_face_numBER, bool bRotateTry)
{
	// 3.1 face detection
	int face_num;
	IplImage *Detect_Image;
	int nNewWidth, nNewHeight;
	double dScale;
	if ((color_image->width>LARGE_IMAGE_SIZE) || (color_image->height>LARGE_IMAGE_SIZE))
	{
		if (color_image->width>color_image->height)
			nNewWidth = STANDARD_IMAGE_WIDTH_LONG;
		else nNewWidth = STANDARD_IMAGE_WIDTH_SMALL;

		dScale = nNewWidth * 1.0 / color_image->width;

		nNewHeight = int(color_image->height * dScale);
		Detect_Image = cvCreateImage(cvSize(nNewWidth, nNewHeight), IPL_DEPTH_8U, color_image->nChannels);
		cvResize(color_image, Detect_Image);
		//		detector.SetFaceDetectionSizeRange(Detect_Image);
		//		detector.SetFaceDetectionROI(Detect_Image, 0.8);

		face_num = detector.detect(Detect_Image, rects, 0);
		//		detector.ClearFaceDetectionRange();
		//detector.ClearFaceDetectionROI();
		for (int i = 0; i<face_num; i++)
		{
			rects[i].rc.x = int(rects[i].rc.x / dScale);
			rects[i].rc.y = int(rects[i].rc.y / dScale);
			rects[i].rc.width = int(rects[i].rc.width / dScale);
			rects[i].rc.height = int(rects[i].rc.height / dScale);
		}
	}
	else
	{
		Detect_Image = cvCloneImage(color_image);
		//	detector.SetFaceDetectionSizeRange(Detect_Image);
		//	detector.SetFaceDetectionROI(Detect_Image, 0.8);
			face_num = detector.detect(Detect_Image, rects, 0);   //for imagelist input
		//	detector.ClearFaceDetectionRange();
		//	//detector.ClearFaceDetectionROI();
	}

	cvReleaseImage(&Detect_Image);

	return face_num;
}

////////////////////////////////////////////////////////////////////////////////////////
/// This routine is a function of the Face Detection Utility
///  written by Ren Haibing 
///   the face model is in the default directory
/// Acknowledge		: modified from Yimin's program
///
/// Function name	: InitFaceDetector
/// Description	    : init face detector 
///
/// Argument		:	
///
/// Return type		: 
///
/// Create Time		: 2014-10-28  14:20
///
///
/// Side Effect		: 
///
///////////////////////////////////////////////////////////////////////////////////////////////
void InitFaceDetector()
{
	tagDetectConfig configParam;
	EnumViewAngle  viewAngle = (EnumViewAngle)VIEW_ANGLE_FRONTAL;
	detector.init(viewAngle, FEA_HAAR, 2);//(EnumFeaType)trackerType);
	//	detector.config( configParam );
}
