#include "stdafx.h"
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(int inNeuronCount, double inEta, double inBatchSize, func inActFunc)
{
	neuronCount = inNeuronCount;
	actFunc = inActFunc;
	eta = inEta;
	batchSize = inBatchSize;
	error = 0;

	correctImages = 0;
	totalImagesRead = 0;
	totalImages = 0;

	epochSize = 60000;
	epochIterator = 1;

	finishedTraining = false;

	x.resize(784);
	w.resize(784);
	deltaw.resize(784);
	z.resize(10);
	derivz.resize(10);
	c.resize(10);
	deltac.resize(10);
	r.resize(10);
	t.resize(10);

	ResizeNeurons();
	InitializeWeights();
}


NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork :: InitializeWeights() {
	random_device rd;
	default_random_engine generator(rd());
	normal_distribution<double> distribution(0, 0.16);

	fill(b.begin(), b.end(), distribution(generator));
	fill(c.begin(), c.end(), distribution(generator));

	for (int i = 0; i < 784; i++) {
		fill(w[i].begin(), w[i].end(), distribution(generator));
	}
	for (int i = 0; i < neuronCount; i++) {
		fill(u[i].begin(), u[i].end(), distribution(generator));
	}
}

void NeuralNetwork::ResizeNeurons() {
	u.resize(neuronCount);
	y.resize(neuronCount);
	s.resize(neuronCount);
	b.resize(neuronCount);
	deltab.resize(neuronCount);
	deltau.resize(neuronCount);
	derivy.resize(neuronCount);

	for (int i = 0; i < 784; i++) {
		w[i].resize(neuronCount);
		deltaw[i].resize(neuronCount);
	}

	for (int i = 0; i < neuronCount; i++) {
		u[i].resize(10);
		deltau[i].resize(10);
	}
}

void NeuralNetwork::InsertInputs(vector<double> inStream) {
	x = inStream;
}

void NeuralNetwork::InsertLabel(vector<double> inStream) {
	t = inStream;
}

void NeuralNetwork::ChangeActivationFunc(func inFunc) {
	actFunc = inFunc;
	InitializeWeights();
}

void NeuralNetwork::ChangeBatchSize(double inBatchSize) {
	batchSize = inBatchSize;
	InitializeWeights();
}

void NeuralNetwork::ChangeNeuronCount(int inCount) {
	neuronCount = inCount;
	ResizeNeurons();
	InitializeWeights();
}

void NeuralNetwork::ChangeEta(double inEta) {
	eta = inEta;
	InitializeWeights();
}

void NeuralNetwork::ChangeImageCount(int ImageCount, int EpochCount) {
	epochSize = ImageCount;
	totalImages = ImageCount * EpochCount;
}

void NeuralNetwork::WriteTestResults(string strOutput) {
	CalculateTestResults();
	std::ofstream outputFile(strOutput, std::ios::in | std::ios::out | std::ios::ate);
	outputFile << "***************************************" << endl;
	outputFile << "Test Results: " << endl;
	outputFile << "Accuracy: " << testError << endl;
	outputFile << "***************************************" << endl;
	outputFile.close();

}

void NeuralNetwork::WriteImageResult(string strOutput) {
	ofstream outputFile(strOutput, std::ios::in | std::ios::out | std::ios::ate);
	outputFile << " Expected Label: " << expectedLabel << " Correct Label: " << correctLabel << endl;
	outputFile.close();
}

void NeuralNetwork:: TestImage() {
	totalImagesRead++;
	finishedTesting = false;

	ResetOuputs();
	CalculateOutput();
	CalculateLabels();
	if (expectedLabel == correctLabel)
		correctImages++;
	
	if (totalImagesRead == totalImages) {
		finishedTesting = true;
	}
}

void NeuralNetwork::TrainImage() {
	totalImagesRead++;
	finishedTraining = false;

	ResetOuputs();
	CalculateOutput();
	CalculateLabels();
	CalculateDeltas();
	BackProp();
	epochIterator = ((int)totalImagesRead / (int)epochSize)+1;

	if (totalImagesRead == totalImages) {
		finishedTraining = true;
		totalImagesRead = 0;
	}
}

void NeuralNetwork::CalculateOutput() {
	for (int i = 0; i < neuronCount; i++) {
		for (int j = 0; j < 784; j++) {
			s[i] += x[j] * w[j][i];
		}
		s[i] += b[i];
		y[i] = ActivationFunction(s[i]);
		derivy[i] = Derivative(y[i]);
	}
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < neuronCount; j++) {
			r[i] += y[j] * u[j][i];
		}
		r[i] += c[i];
		z[i] = ActivationFunction(r[i]);
		error += pow((z[i] - t[i]), 2);
		derivz[i] = Derivative(z[i]);
	}
	error = error / 10.0;
}

void NeuralNetwork::CalculateDeltas() {
	bool calculatedB = false;
	bool calculatedU = false;
	bool calculatedC = false;
	bool uIsInitialized = false;
	vector<double> placeHolderW(neuronCount);
	fill(deltac.begin(), deltac.end(), 0);
	fill(deltab.begin(), deltab.end(), 0);

	for (int i = 0; i < neuronCount; i++) {
		for (int j = 0; j <10; j++) {
			//placeHolder will represent summation for the weight
			placeHolderW[i] += (z[j] - t[j]) *(derivz[j])*u[i][j];
			deltab[i] += (z[j] - t[j]) *(derivz[j]) *u[i][j];
			deltau[i][j] = (z[j] - t[j])*(derivz[j] * y[i]);
		}
		deltab[i] = deltab[i] / 10.0;
	}

	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < neuronCount; j++) {
			deltaw[i][j] = placeHolderW[j] / 10.0;
			deltaw[i][j] *= derivy[j] * x[i];
		}
	}

	for (int i = 0; i < 10; i++) {
		deltac[i] = (z[i] - t[i])*derivz[i];
	}

}

void NeuralNetwork::CalculateLabels() {
	double maxValue = -100;
	for (int i = 0; i < 10; i++) {
		if (maxValue < z[i]) {
			maxValue = z[i];
			expectedLabel = i;
		}
		if (t[i] == 1) {
			correctLabel = i;
		}
	}
}

void NeuralNetwork::CalculateTestResults() {
	testError = (correctImages / totalImages) * 100;
}


void NeuralNetwork::BackProp() {
	bool calculatedB = false;
	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < neuronCount; j++) {
			w[i][j] = w[i][j] - (eta * deltaw[i][j]);
			if (!calculatedB)
				b[j] = b[j] - (eta * deltab[j]);
		}
		calculatedB = true;
	}
	bool calculatedC = false;
	for (int i = 0; i < neuronCount; i++) {
		for (int j = 0; j < 10; j++) {
			u[i][j] = u[i][j] - (eta * deltau[i][j]);
			if (!calculatedC)
				c[j] = c[j] - (eta * deltac[j]);
		}
		calculatedC = true;
	}
}

void NeuralNetwork::DisplayError() {
	cout << " Accuracy: " << (1.0-error)*100 << "%" << endl; 
}

void NeuralNetwork::DisplayLabels() {
	cout << " Expected Label: " << expectedLabel << " Correct Label: " << correctLabel << endl;
}

double NeuralNetwork::GetAccuracy() {
	CalculateTestResults();
	return testError;
}

int NeuralNetwork :: GetCorrectImages() {
	return correctImages;
}


double NeuralNetwork::ActivationFunction(double inValue) {
	switch (actFunc) {
	case TANH:
		return tanh(inValue);
	case SIGM:
		return (inValue) / (1.0 + abs(inValue));
	case RELU:
		if (inValue > 0)
			return inValue;
		else
			return 0;
	case DBLSIG:
		return ((2.0 / (1 + exp(-(inValue)))) - 1.0);
		break;
	default:
		return tanh(inValue);;
	}
}

double NeuralNetwork::Derivative(double inValue) {
	switch (actFunc) {
	case TANH:
		return 1.0 - pow(inValue, 2);
	case SIGM:
		return (inValue * (1.0 - inValue));
	case RELU:
		if (inValue > 0)
			return 1.0;
		else
			return 0.0;
	case DBLSIG:
		return ((1.0 - inValue) * (1.0 + inValue));
	default:
		return 0;
	}
}

void NeuralNetwork::ResetOuputs() {
	error = 0;
	fill(r.begin(), r.end(), 0);
	fill(z.begin(), z.end(), 0);
	fill(y.begin(), y.end(), 0);
	fill(s.begin(), s.end(), 0);
}

bool NeuralNetwork::FinishedTraining() {
	return finishedTraining;
}

bool NeuralNetwork::FinishedTesting() {
	return finishedTesting;
}