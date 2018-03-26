#include "stdafx.h"
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(int inNeuronCount, double inEta, int inBatchSize, func inActFunc)
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

	images.resize(inBatchSize);
	labels.resize(inBatchSize);

	//resize the batchDelta values
	deltawBatch.resize(inBatchSize);
	deltabBatch.resize(inBatchSize);
	deltauBatch.resize(inBatchSize);
	deltacBatch.resize(inBatchSize);


	//resize first layer attributess
	w.resize(784);
	deltaw.resize(784);

	//Resize second layer attributes
	z.resize(10);
	derivz.resize(10);
	c.resize(10);
	deltac.resize(10);
	r.resize(10);

	//Resize constant second Layer DeltaC Batches
	for (unsigned int i = 0; i < batchSize; i++) {
		deltacBatch[i].resize(10);
	}

	ResizeNeurons();
	ResizeDeltaBatches();
	ResizeImages();
	ResizeLabels();
	InitializeWeights();
}


NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork :: InitializeWeights() {
	random_device rd;
	default_random_engine generator(rd());
	normal_distribution<double> distribution(0, 0.16);

	for (int i = 0; i < neuronCount; i++) {
		for (int j = 0; j < 10; j++) {
			u[i][j] = distribution(generator);
		}
	}

	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < neuronCount; j++) {
			w[i][j] = distribution(generator);
		}
	}
	for (int i = 0; i < b.size(); i++) {
		b[i] = distribution(generator);
	}
	for (int i = 0; i < c.size(); i++) {
		c[i] = distribution(generator);
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

void NeuralNetwork::ResizeImages() {
	for (unsigned int i = 0; i < batchSize; i++) {
		images[i].resize(784);
	}
}

void NeuralNetwork::ResizeLabels() {
	for (unsigned int i = 0; i < batchSize; i++) {
		labels[i].resize(10);
	}
}

void NeuralNetwork::ResizeDeltaBatches() {
	//Resize all the Delta 
	for (unsigned int i = 0; i < batchSize; i++) {
		deltawBatch[i].resize(784);
		deltabBatch[i].resize(neuronCount);
		deltauBatch[i].resize(neuronCount);
		for (unsigned int j = 0; j < 784; j++) {
			deltawBatch[i][j].resize(neuronCount);
		}
		for (unsigned int j = 0; j < neuronCount; j++) {
			deltauBatch[i][j].resize(10);
		}
	}
}

void NeuralNetwork::InsertInputs(vector<double> inStream) {
	int counter = 0;
	for (unsigned int i = 0; i < batchSize; i++) {
		for (unsigned int j = 0; j < 784; j++) {
			images[i][j] = inStream[counter];
			counter++;
		}
	}
}

void NeuralNetwork::InsertLabels(vector<double> inStream) {
	unsigned int counter = 0;
	for (unsigned int i = 0; i < batchSize; i++) {
		for (unsigned int j = 0; j < 10; j++) {
			labels[i][j] = inStream[counter];
			counter++;
		}
	}
}

void NeuralNetwork::ChangeActivationFunc(func inFunc) {
	actFunc = inFunc;
	InitializeWeights();
}

void NeuralNetwork::ChangeBatchSize(unsigned int inBatchSize) {
	batchSize = inBatchSize;
	InitializeWeights();
}

void NeuralNetwork::ChangeNeuronCount(unsigned int inCount) {
	neuronCount = inCount;
	ResizeNeurons();
	InitializeWeights();
}

void NeuralNetwork::ChangeEta(double inEta) {
	eta = inEta;
	InitializeWeights();
}

void NeuralNetwork::ChangeImageCount(unsigned int ImageCount, unsigned int EpochCount) {
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
	CalculateOutput(0);
	CalculateLabels(0);

	if (expectedLabel == correctLabel)
		correctImages++;
	
	if (totalImagesRead == totalImages) {
		finishedTesting = true;
		totalImagesRead = 0;
	}
}

void NeuralNetwork::SetForTest() {
	batchSize = 1;
	totalImages = 0;
	finishedTesting = false;
	correctImages = 0;
}

void NeuralNetwork::TrainImage() {
	finishedTraining = false;

	for (unsigned int i = 0; i < batchSize; i++) {
		totalImagesRead++;
		ResetOuputs();
		CalculateOutput(i);
		CalculateLabels(i);
		CalculateDeltaBatch(i);
	}
	CalculateDeltaAverages();
	BackProp();
	epochIterator = ((int)totalImagesRead / (int)epochSize)+1;

	if (totalImagesRead == totalImages) {
		finishedTraining = true;
		totalImagesRead = 0;
	}
}

void NeuralNetwork::CalculateOutput(int index) {
	for (unsigned int i = 0; i < neuronCount; i++) {
		for (unsigned int j = 0; j < 784; j++) {
			s[i] += images[index][j] * w[j][i];
		}
		s[i] += b[i];
		y[i] = ActivationFunction(s[i]);
		derivy[i] = Derivative(y[i]);
	}
	for (unsigned int i = 0; i < 10; i++) {
		for (unsigned int j = 0; j < neuronCount; j++) {
			r[i] += y[j] * u[j][i];
		}
		r[i] += c[i];
		z[i] = ActivationFunction(r[i]);
		error += pow((z[i] - labels[index][i]), 2);
		derivz[i] = Derivative(z[i]);
	}
	error = error / 10.0;
}

void NeuralNetwork::CalculateDeltaBatch(int index) {
	bool calculatedB = false;
	bool calculatedU = false;
	bool calculatedC = false;
	bool uIsInitialized = false;
	int inIndex = index;
	vector<double> placeHolderW(neuronCount);
	fill(deltac.begin(), deltac.end(), 0);
	fill(deltab.begin(), deltab.end(), 0);

	for (unsigned int i = 0; i < neuronCount; i++) {
		for (unsigned int j = 0; j <10; j++) {
			//placeHolder will represent summation for the weight
			placeHolderW[i] += (z[j] - labels[index][j]) *(derivz[j])*u[i][j];
			deltabBatch[index][i] += (z[j] - labels[index][j]) *(derivz[j]) *u[i][j];
			deltauBatch[index][i][j] = (z[j] - labels[index][j])*(derivz[j] * y[i]);
		}
		deltabBatch[index][i] = deltab[i] / 10.0;
	}

	for (unsigned int i = 0; i < 784; i++) {
		for (unsigned int j = 0; j < neuronCount; j++) {
			deltawBatch[index][i][j] = placeHolderW[j] / 10.0;
			deltawBatch[index][i][j] *= derivy[j] * images[index][i];
		}
	}

	for (unsigned int i = 0; i < 10; i++) {
		deltacBatch[index][i] = (z[i] - labels[index][i])*derivz[i];
	}
}

void NeuralNetwork::CalculateDeltaAverages() {
	vector<vector<double>> deltawSum(784);
	vector<vector<double>> deltauSum(neuronCount);
	vector<double> deltacSum(10);
	vector<double> deltabSum(neuronCount);

	//resize the sum vectors
	for (unsigned int i = 0; i < 784; i++) {
		deltawSum[i].resize(neuronCount);
	}
	for (unsigned int i = 0; i < neuronCount; i++) {
		deltauSum[i].resize(10);
	}

	for (unsigned int i = 0; i < 784; i++) {
		for (unsigned int j = 0; j < neuronCount; j++) {
			for (unsigned int l = 0; l < batchSize; l++) {
				deltawSum[i][j] += deltawBatch[l][i][j];
			}
		}
	}

	for (unsigned int i = 0; i < neuronCount; i++) {
		for (unsigned int j = 0; j < 10; j++) {
			for (unsigned int l = 0; l < batchSize; l++) {
				deltauSum[i][j] += deltauBatch[l][i][j];
			}
		}
	}

	for (unsigned int i = 0; i < neuronCount; i++) {
		for (unsigned int j = 0; j < batchSize; j++) {
			deltabSum[i] += deltabBatch[j][i];
		}
	}

	for (unsigned int i = 0; i < 10; i++) {
		for (unsigned int j = 0; j < batchSize; j++) {
			deltacSum[i] += deltacBatch[j][i];
		}
	}

	//Find the average
	for (unsigned int i = 0; i < 784; i++) {
		for (unsigned int j = 0; j < neuronCount; j++) {
			deltaw[i][j] = deltawSum[i][j] / batchSize;
		}
	}

	for (unsigned int i = 0; i < neuronCount; i++) {
		for (unsigned int j = 0; j < 10; j++) {
			deltau[i][j] = deltauSum[i][j] / batchSize;
		}
	}

	for (unsigned int i = 0; i < neuronCount; i++) {
		deltab[i] = deltabSum[i] / batchSize;
	}

	for (unsigned int i = 0; i < 10; i++) {
		deltac[i] = deltacSum[i] / batchSize;
	}
}

void NeuralNetwork::CalculateLabels(int index) {
	double maxValue = -100;
	for (unsigned int i = 0; i < 10; i++) {
		if (maxValue < z[i]) {
			maxValue = z[i];
			expectedLabel = i;
		}
		if (labels[index][i] == 1) {
			correctLabel = i;
		}
	}
}

void NeuralNetwork::CalculateTestResults() {
	testError = (correctImages / totalImages) * 100;
}


void NeuralNetwork::BackProp() {
	bool calculatedB = false;
	for (unsigned int i = 0; i < 784; i++) {
		for (unsigned int j = 0; j < neuronCount; j++) {
			w[i][j] = w[i][j] - (eta * deltaw[i][j]);
			if (!calculatedB)
				b[j] = b[j] - (eta * deltab[j]);
		}
		calculatedB = true;
	}
	bool calculatedC = false;
	for (unsigned int i = 0; i < neuronCount; i++) {
		for (unsigned int j = 0; j < 10; j++) {
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

int NeuralNetwork::GetExpectedLabel() {
	return expectedLabel;
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