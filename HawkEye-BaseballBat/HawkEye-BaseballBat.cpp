/*Hawkeye "Baseball bat" detector using "Darknet", YOLOv4 and OpenCV 4.3.0 by lewis lewin*/
///
///
///			TO DO LIST
/// 
/// 3. Create working main class including a loop with proper options
/// 4. Add some diagnostic information eg time taken to process photo how many objects found in photo etc
/// 6 Add xml / json output
/// 
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <filesystem>
#include <algorithm>
#define OPENCV
#include <yolo_v2_class.hpp>
#include <opencv2/opencv.hpp>
namespace fs = std::filesystem;

class ObjectDetector {
	/// <summary>
	/// Object detection class
	/// used for detecting objects in image files
	/// parse() is main function
	/// </summary>
	
	struct file
	{
		/// <summary>
		/// 
		/// </summary>
		bool found = 0;
		std::vector<bbox_t> allObjects;
		std::vector<bbox_t> objects;
		cv::Mat image;
		std::string path;
		std::string filename;
		bool rotated = 0;
	};

	int objectId = -1;
	std::string cfgFile = "yolov4.cfg";
	std::string weightsFile = "yolov4.weights";
	
	std::vector<file> imageVec;
	
public:
	int parse(std::string directory, int objectId = -1)
	{	/// <summary>
		/// Takes the directory of images and the object ID of an object that is being searched for, initiates the detector and feeds it the images
		/// </summary>
		/// <param name="directory"></param>
		/// <param name="objectId"></param>
		/// <returns>0, no error, 1 failed to parse</returns>
		//fill imagevec with files
		fetchImage(directory);

		Detector detector(cfgFile, weightsFile);
		//first detection origional image
		for (int i = 0; i < imageVec.size(); i++) {
			imageVec[i].allObjects = detector.detect(imageVec[i].image);
			std::cout << "Processed: " << imageVec[i].filename << std::endl;
			if (!(objectId == -1)) {
				for (int j = 0; j < imageVec[i].allObjects.size(); j++) {
					if (imageVec[i].allObjects[j].obj_id == objectId) {
						imageVec[i].objects.push_back(imageVec[i].allObjects[j]);
						imageVec[i].found = 1;
						std::cout << "Found object on first pass in: " << imageVec[i].filename << std::endl;
					}
				}
			}
			if (!imageVec[i].objects.size()) {
				
				
				std::cout << "Object not found in : " << imageVec[i].filename << " on first pass, Rotating 90 and reprocessing..." << std::endl;
				cv::rotate(imageVec[i].image, imageVec[i].image, cv::ROTATE_90_CLOCKWISE);
				imageVec[i].rotated = 1;
				imageVec[i].allObjects = detector.detect(imageVec[i].image);
				
				
				for (int j = 0; j < imageVec[i].allObjects.size(); j++) {
					if (imageVec[i].allObjects[j].obj_id == objectId) {
						imageVec[i].objects.push_back(imageVec[i].allObjects[j]);
						imageVec[i].found = 1;
						std::cout << "Found object on second pass in: " << imageVec[i].filename << std::endl;
					}
				}


				for (int p = 0; p < imageVec[p].objects.size(); p++) {
					imageVec[i].objects[p] = correctBoundingBox(imageVec[i].objects[p], imageVec[i].image);
				}
				cv::rotate(imageVec[i].image, imageVec[i].image, cv::ROTATE_90_COUNTERCLOCKWISE);
				
			}
		}
		return 0;
	}
	void displayBestObject() {
		for (int i = 0; i < imageVec.size(); i++) {
			if (imageVec[i].found) {
				auto temp = bestMatch(imageVec[i].objects);
				cv::Scalar color(60, 160, 260);
				cv::rectangle(imageVec[i].image, cv::Rect(temp.x, temp.y, temp.w, temp.h), color, 3);
				cv::imshow(imageVec[i].filename, imageVec[i].image);
				cv::waitKey(0);
			}
			else {
				cv::imshow(imageVec[i].filename, imageVec[i].image);
			}
		}
	}
	void writeBestObject() {
		std::ofstream results;
		results.open(imageVec[0].path + "/results.txt", std::ios::out | std::ios::trunc);
		for (int i = 0; i < imageVec.size(); i++) {
			if (imageVec[i].found) {
				auto temp = bestMatch(imageVec[i].objects);
				cv::Scalar color(60, 160, 260);
				cv::rectangle(imageVec[i].image, cv::Rect(temp.x, temp.y, temp.w, temp.h), color, 3);
				cv::imwrite(imageVec[i].path + "/res_" + imageVec[i].filename, imageVec[i].image);
				results << "Found object in file: " << imageVec[i].filename << " Object located in zone: (X1 = " << temp.x << ", x2 = " << temp.x + temp.w << ", Y1 = " << temp.y << ", Y2 = " << temp.y + temp.h << ")" << std::endl;
				cv::waitKey(0);
			}
			else {
				results << "Unable to find object in file: " << imageVec[i].filename << std::endl;
			}
			
		}
		results.close();
	}
private:
	cv::Point getCentre(cv::Mat img)
	{
		cv::Point centre;
		centre.x = img.cols / 2;
		centre.y = img.rows / 2;
		return centre;
	}
	bbox_t bestMatch(std::vector<bbox_t> rVec) {
		/// <summary>
		/// Returns vector element with highest probability of being searched for object
		/// </summary>
		/// <param name="rVec"><bbox_t> vector to be sorted through</param>
		/// <returns></returns>
		auto temp = rVec[0];
		for (int i = 1; i < rVec.size(); i++) {
			if (temp.prob < rVec[i].prob)
				temp = rVec[i];
		}
		return temp;
	}
	int fetchImage(std::string directory) {
		for (const auto& entry : fs::directory_iterator(directory)) {
			
			std::string tempFullPath;
			file tempImageStruct;

			auto i = entry.path();
			tempFullPath = i.string();

			//split temp into path and filename
			std::size_t found = tempFullPath.find_last_of("/\\");
			tempImageStruct.path = tempFullPath.substr(0, found);
			tempImageStruct.filename = tempFullPath.substr(found + 1);
			//check file is image and continue if not
			std::size_t	ext = tempImageStruct.filename.find_last_of(".");
			if (!((tempImageStruct.filename.substr(ext + 1) == "png") || tempImageStruct.filename.substr(ext + 1) == "jpg" || tempImageStruct.filename.substr(ext + 1) == "jpeg") || tempImageStruct.filename.substr(ext + 1) == "bmp")
				continue;
			//convert files to matrixes and store in imageVec
			tempImageStruct.image = cv::imread(tempFullPath);
			imageVec.push_back(tempImageStruct);
		}
		return 0;
	}
	bbox_t correctBoundingBox(bbox_t boundingBox, cv::Mat image) {
		bbox_t rotated;
		//find boundingbox x,y dist from centre
		rotated = boundingBox;
		rotated.x = boundingBox.y;
		rotated.y = image.cols - boundingBox.x;
		//swap x,y and change sign of y
		rotated.y = rotated.y - boundingBox.w;
		rotated.w = boundingBox.h;
		rotated.h = boundingBox.w;
		return rotated;
	}
};



int main() {


	while (true) {
		std::string dir;
		int objID;
		std::cout << "Please enter the directory of images you wish to use or leave (\"exit\") to quit: " << std::endl;
		std::cin >> dir;
		if (dir == "Exit")
			break;
		if (dir == "exit")
			break;
		std::cout << "Please input the ID of an object you wish to detect: " << std::endl;
		std::cin >> objID;
		ObjectDetector det;
		det.parse(dir, objID);
		det.displayBestObject();
		det.writeBestObject();
	}
	return 0;
}