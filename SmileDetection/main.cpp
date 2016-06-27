#include"SmileDetection.h"
#include"RSWrapper.h"

int main(int argc, char* argv[])
{
	bool simleFlag = false;
	int face_num = 0;
	RSWrapper myRSWrapper;
	myRSWrapper.init();
	Mat rgbImage, depthImage;
	char imageName[200];
	int count = 0;
	while(myRSWrapper.capture(0, rgbImage, depthImage, 0))
	{
		IplImage color_image = rgbImage;
		CvRectItem rects[MAX_FACE_NUMBER];
		InitFaceDetector();
		face_num = FaceDetectionApplication(&color_image, rects, MAX_FACE_NUMBER, false);
		simleFlag = FaceRecognitionApplication(&color_image, face_num, rects);	
		imshow("Image", rgbImage); cvWaitKey(1);
		if (simleFlag)
		{
		//	cout << "Detect someone was laughing"<<'\r';
			sprintf_s(imageName, "./SimleImage/Smile_%04d.jpg", count++);                                                        
			imwrite(imageName, rgbImage);
		}
		
	}
	
	return 0;
}