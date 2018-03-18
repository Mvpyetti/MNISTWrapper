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
		void SetEpochCount(double);
		void SetEta(double);
		void SetActFunc(System::String^);
		void TrainNetwork();
		void TestNetwork();
		bool ReadImages(System::String^);
		bool ReadLabels(System::String^);
		int GetImagesRead();
		int GetTotalImages();
		int GetEpochIterator();
		int GetEpochSize();
		bool FinishedTraining();
		bool FinishedReadingImages();
		bool FinishedReadingLabels();

	private:
		NeuralNetwork * NN;
		MNIST * dataFile;
		double * epochCount;
		map<string, func> * functionMapper;
	};
}
