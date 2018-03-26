#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <vector> 
using namespace std;

class MNIST
{
public:
	MNIST();
	~MNIST();
	bool ReadInputFile(string);
	bool ReadLabelFile(string);
	vector<double> GetImage();
	vector<double> GetImage(int);
	vector<double> GetImages(unsigned int);
	vector<double> GetLabel();
	vector<double> GetLabel(int);
	vector<double> GetLabels(unsigned int);
	int GetNumOfImages();
	void PrintStats();
	void ResetMNIST();

	bool finishedReadingInputs;
	bool finishedReadingLabels;

private:
	vector<double> ConvertVector(int);
	int FindBinaryValue(int);

	int data_magic_number;
	int label_magic_number;
	int num_of_images;
	int num_of_labels;
	int Number_of_Rows;
	int Number_of_Columns;
	int lastImgIndex;
	int lastlblIndex;
	vector<double> images;
	vector<double> labels;
};

