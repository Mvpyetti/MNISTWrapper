// MnistWrapper.h

#pragma once
#include "MNIST.h"
#include "NeuralNetwork.h"
#include <string>
#include <map>

using namespace System;
using namespace System::Runtime::InteropServices;


namespace MnistWrapper {

	public ref class MnistWrapperClass
	{
	public:
		MnistWrapperClass();
		void SetNeuronCount(int);
		void SetBatchSize(int);
		void SetEpochCount(double);
		void SetEta(double);
		void SetActFunc(System::String^);
		void TrainNetwork();
		void TestNetwork();
		void TestRandomImage(int);
		bool ReadImages(System::String^);
		bool ReadLabels(System::String^);
		void ResetMNIST();
		double GetAccuracy();
		int GetCorrectImages();
		int GetExpectedLabel();
		int GetImagesRead();
		int GetTotalImages();
		int GetEpochIterator();
		int GetEpochSize();
		bool FinishedTraining();
		bool FinishedTesting();
		bool FinishedReadingImages();
		bool FinishedReadingLabels();
		void WriteOutput();

	private:
		NeuralNetwork * NN;
		MNIST * dataFile;
		double * epochCount;
		int batchSize;
		map<string, func> * functionMapper;
	};
}
