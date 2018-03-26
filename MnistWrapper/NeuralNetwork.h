#pragma once
#include <vector>
#include <math.h>
#include <random>
#include <fstream>
#include <iostream>

using namespace std; 

enum func{
	TANH,
	SIGM,
	DBLSIG,
	RELU
};

class NeuralNetwork
{
public:
	NeuralNetwork(int = 50, double = .005, int = 5, func = TANH);
	void ChangeActivationFunc(func);
	void ChangeBatchSize(unsigned int);
	void ChangeImageCount(unsigned int, unsigned int);
	void ChangeNeuronCount(unsigned int);
	void ChangeEta(double);
	void DisplayError();
	void DisplayLabels();
	void InsertInputs(vector<double>);
	void InsertLabels(vector<double>);
	void TrainImage();
	void TrainBatch();
	void TestImage();
	void SetForTest();
	void WriteTestResults(string);
	void WriteImageResult(string);
	~NeuralNetwork();

	double GetAccuracy();
	int GetExpectedLabel();
	int GetCorrectImages();

	bool FinishedTraining();
	bool FinishedTesting();

	double totalImages;
	double totalImagesRead;
	double correctImages;
	double epochIterator;
	double epochSize;
	double batchSize;
private:
	void BackProp();
	void CalculateOutput(int);
	void CalculateDeltaBatch(int);
	void CalculateDeltaAverages();
	void CalculateLabels(int);
	void CalculateTestResults();
	void InitializeWeights();
	void ResizeLabels();
	void ResizeNeurons();
	void ResizeImages();
	void ResizeDeltaBatches();
	void ResetOuputs();

	double ActivationFunction(double);
	double Derivative(double);


	//Images array
	vector<vector<double>> images;
	vector<vector<double>> labels;

	//STANDARD VALUES
	vector<vector<double>> w;
	vector<double> b;
	vector<double> s;
	vector<double> y;
	vector<double> derivy;
	vector<vector<double>> u;
	vector<double> c;
	vector<double> r;
	vector<double> z;
	vector<double> derivz;

	//DELTA VALUES
	vector<vector<double>> deltaw;
	vector<double> deltab;
	vector<vector<double>> deltau;
	vector<double> deltac;

	//Delta Batch Values
	vector<vector<vector<double>>> deltawBatch;
	vector<vector<double>> deltabBatch;
	vector<vector<vector<double>>> deltauBatch;
	vector<vector<double>> deltacBatch;

	//HYPER PARAMETERS
	func actFunc;
	int neuronCount;
	double eta;

	//Testing Variables
	int expectedLabel;
	int correctLabel;
	double testError;
	double error;

	bool finishedTraining;
	bool finishedTesting;
};

