// This is the main DLL file.

#include "stdafx.h"

#include "MnistWrapper.h"

MnistWrapper::MnistWrapperClass::MnistWrapperClass() {
	NN = new NeuralNetwork();
	dataFile = new MNIST;
	epochCount = new double;

		//generate the mapper for enumerators
	functionMapper = new map<string, func>;
	functionMapper->insert(pair<string, func>("TANH", TANH));
	functionMapper->insert(pair<string, func>("SIGMOID", SIGM));
	functionMapper->insert(pair<string, func>("DOUBLE SIG", DBLSIG));
	functionMapper->insert(pair<string, func>("ReLU", RELU));

}

void MnistWrapper::MnistWrapperClass::SetActFunc(System::String^ inFunc) {	
	//NOTE: We have to use this weird system string so that C# can pass in string values
	char* charPtr = (char*)Marshal::StringToHGlobalAnsi(inFunc).ToPointer();
	string strInFunc(charPtr);
	func enumFunc = (*functionMapper)[strInFunc];
	NN->ChangeActivationFunc(enumFunc);
	Marshal::FreeHGlobal(IntPtr(charPtr));
}

void MnistWrapper::MnistWrapperClass::SetBatchSize(int inBatchSize) {
	batchSize = inBatchSize;
}

void MnistWrapper::MnistWrapperClass::SetEpochCount(double inEpoch) {
	*epochCount = inEpoch;
}

void MnistWrapper::MnistWrapperClass::SetEta(double inEta) {
	NN->ChangeEta(inEta);
}

void MnistWrapper::MnistWrapperClass::SetNeuronCount(int inNeuronCount) {
	NN->ChangeNeuronCount(inNeuronCount);
}

void MnistWrapper::MnistWrapperClass::TrainNetwork() {
	for (int i = 0; i < *epochCount; i++) {
		for (int j = 0; j < dataFile->GetNumOfImages()/batchSize; j++) {
			NN->InsertInputs(dataFile->GetImages(batchSize));
			NN->InsertLabels(dataFile->GetLabels(batchSize));
			NN->TrainImage();
		}
	}
}

void MnistWrapper::MnistWrapperClass::TestNetwork() {
	NN->SetForTest();
	for (int i = 0; i < dataFile->GetNumOfImages(); i++) {
		NN->InsertInputs(dataFile->GetImage());
		NN->InsertLabels(dataFile->GetLabel());
		NN->TestImage();
	}
}

void MnistWrapper::MnistWrapperClass::TestRandomImage(int index) {
	NN->InsertInputs(dataFile->GetImage(index));
	NN->InsertLabels(dataFile->GetLabel(index));
	NN->TestImage();
}

bool MnistWrapper::MnistWrapperClass::ReadImages(System::String^ strFile) {
	char* charPtr = (char*)Marshal::StringToHGlobalAnsi(strFile).ToPointer();
	string cstrFile = string(charPtr);
	Marshal::FreeHGlobal(IntPtr(charPtr));
	if (dataFile->ReadInputFile(cstrFile)) {
		NN->ChangeImageCount(dataFile->GetNumOfImages(), *epochCount);
		return true;
	}
	else
		return false;
}

bool MnistWrapper::MnistWrapperClass::ReadLabels(System::String^ strFile) {
	char* charPtr = (char*)Marshal::StringToHGlobalAnsi(strFile).ToPointer();
	string cstrFile = string(charPtr);
	Marshal::FreeHGlobal(IntPtr(charPtr));
	if (dataFile->ReadLabelFile(cstrFile))
		return true;
	else
		return false;
}

void MnistWrapper::MnistWrapperClass::ResetMNIST() {
	dataFile->ResetMNIST();
}

bool MnistWrapper::MnistWrapperClass:: FinishedTraining() {
	return NN->FinishedTraining();
}

bool MnistWrapper::MnistWrapperClass::FinishedTesting() {
	return NN->FinishedTesting();
}

bool MnistWrapper::MnistWrapperClass::FinishedReadingImages() {
	return dataFile->finishedReadingInputs;
}

bool MnistWrapper::MnistWrapperClass::FinishedReadingLabels() {
	return dataFile->finishedReadingLabels;
}

double MnistWrapper::MnistWrapperClass::GetAccuracy() {
	return NN->GetAccuracy();
}

int MnistWrapper::MnistWrapperClass::GetCorrectImages() {
	return NN->GetCorrectImages();
}

int MnistWrapper::MnistWrapperClass::GetExpectedLabel() {
	return NN->GetExpectedLabel();
}

int MnistWrapper::MnistWrapperClass::GetTotalImages() {
	return NN->totalImages;
}

int MnistWrapper::MnistWrapperClass::GetImagesRead() {
	return NN->totalImagesRead;
}

int MnistWrapper::MnistWrapperClass::GetEpochIterator() {
	return NN->epochIterator;
}

int MnistWrapper::MnistWrapperClass::GetEpochSize() {
	return NN->epochSize;
}

void MnistWrapper::MnistWrapperClass::WriteOutput() {
	string output = "OutputResults";
	NN->WriteTestResults(output);
}