#include "Network.h"

Network::Network()
{
	//����������� ���������, ��������� ������� ������� � ��������, ����������� ���������� �������
	this->weights1 = initMatrix(INPUT_DIM, HIDDEN_DIM);
	this->bias1 = initMatrix(1, HIDDEN_DIM);
	this->weights2 = initMatrix(HIDDEN_DIM, OUT_DIM);
	this->bias2 = initMatrix(1, OUT_DIM);
}

Network::~Network()
{
	//����������, ����������� ������, ������ ��� ��������� �������
	deleteMatrix(this->weights1);
	deleteMatrix(this->bias1);
	deleteMatrix(this->weights2);
	deleteMatrix(this->bias2);
	deleteMatrix(this->losses);
}

double* Network::relu(double* array, int n)
{
	/*
	* ������� ��������� relu, ����������� ��� (x, ���� x > 0, ����� 0), ����� �����������, ��� ����� ��� ��������
	* ������� ����� �������� �� ����� ������, �������� �������, ����� ������� ���������� � ����������� �� ���� ������� (����� drelu)
	* ������� ��������� ��������� �������� ������ �� ���������� ������ � ����������� ��� � ��������� �����,
	* ������� ����� ������������ � �������� ������� ������ ��� ��������� ������.
	* ������� ��������� ��������� ������������ � ��������� ����, ��� ��������� �� ������ ����� ������� "������"
	*/
	
	//�������� ��������� ������� ��� ������� �������� ��������
	double* outArray = new double[n];
	//���������� ������� ��������� � ������� �������� �������� ������� � ������ � �������� ������
	for (int i = 0; i < n; i++)
	{
		outArray[i] = array[i] >= 0 ? array[i] : 0;
	}
	//����������� ��������� �� �������� ������
	return outArray;
}

double* Network::softmax(double* array, int n)
{
	/*
	* ������� ��������� softmax, ����������� ��� (ei^x / sum(ej^x), ��� j �������� �� 0 �� n, � i ��� ������ �������� �������), 
	* ��� ������� �������� �������������� ���������� � ������� �������� ��������, ������� �� ����� ��������� � ������� �������� ��������� ����� �������� �������,
	* �������� � �������, ������������ ������ (������ �����) � �����������, ������ ����� ��������� ����� �������.
	* ������� ������� � �������� ���������� ������, �������������� ������������� ������������ ������ ������������� �����������.
	* ������ ������� ������� �������� ��� ������ �������������
	*/

	//�������� ��������� ������� ��� ������� �������� ��������
	double* outArray = new double[n];
	
	//����� ���� ��������� � ������� ���������
	double summa = 0;

	//������ �� ����� �������
	for (int i = 0; i < n; i++)
	{
		//���������� ���������� � ������� ��������
		double item = exp(array[i]);
		//������ � �������� ������
		outArray[i] = item;
		//����������� � �����
		summa += item;
	}

	//������ �� ����� �������
	for (int i = 0; i < n; i++)
	{
		//���������� ��������� �������� ��� ������� ��������, ��������� � ������� ��������, ������� �� ����� ����� ���������
		outArray[i] = outArray[i] / summa;
	}

	//����������� ��������� �� �������� ������
	return outArray;
}


double Network::drelu(double x)
{
	/*
	* ������������ �� ������� ��������� relu, ����������� ��� (1, ���� x > 0, ����� 0)
	* ���������� ��� �������� ���������
	*/
	return x >= 0 ? 1. : 0.;
}

double Network::sparse_cross_entropy(double* array, int value)
{
	/*
	* 
	* ������� ���������� ������, ��������������-�����-��������
	* ������������ ��� �������� ���������
	* ������� �������� ������ ����� ���� ������, �� ��� ������ ������������� ����� ����� �������� ������ ���
	* ������� �������� � ����������� ����, ��� ��������� ������ �������� �����, ������� �������� ������ ���������� � ���������
	*/
	return -log(array[value]);
}

double* Network::predict(double* data)
{
	/*
	* ����� ������������, ������ ���
	* �������������� ������� ��������� �������, ����� � ��� ���� 2 ������� ������� (N1, N2) � 2 �������� (T1, T2), �����
	* ��� ����������� ���� (W11 �� N1 � T1, W12 �� N1 � T2, W21 �� N2 � T1, W22 �� N2 � T2), � ����� �������� (bias1 ��� T1 � bias2 ��� T2)
	* ������� �������� �� �������� �������� ����� ����� ���:
	* T1 = N1*w11 + N2*w21 + bias1
	* T2 = N1*w12 + N2*w22 + bias2
	* ������ �������� ����� ����������� � ��������� ����:
	* ����� � ��� ���� ������� �����
	* W (2x2) =
	* (w11, w12,
	*  w21, w22)
	* ������� ������� �������� (1x2) N = (N1, N2)
	* ������� �������� (1x2) B = (bias1, bias2)
	* ����� ������� �������� �������� (1x2) OUT ����� �������� ���:
	* OUT = N * W + B
	* ��������� ������ ���������� ������� ��������� ��� ������� �� ��������� OUT
	* ��� ���� ��������� ���������� ���������� ���� �������� �� ������ �����������, ��������� ���� ����� ������������ OUT ��� �������
	*/
	//��������� ������ ��� �������� �������� �� ������� ���� ��������
	double* out = new double[HIDDEN_DIM];

	//������������ ������ (data), ����������� (1x784) �� ������� ����� (weights1), ����������� (784x128) � ������ ���������� � ������ out
	multiplyMatrix(1, HIDDEN_DIM, INPUT_DIM, data, this->weights1, out);
	
	//�������� ������������ �������� �� �������� (out) �� ���������� (bias1), ������ � out
	sumMatrix(1, HIDDEN_DIM, out, this->bias1, out);

	//���������� ������� ��������� ��� ������� ������� OUT, �������� ����� ������ fout
	double* fout = this->relu(out, HIDDEN_DIM);



	//��������� ������ ��� �������� �������� �� �������� ���� ��������
	double* out2 = new double[OUT_DIM];

	//������������ ������ (fout), ����������� (1x128) �� ������� ����� (weights2), ����������� (128x10) � ������ ���������� � ������ out2
	multiplyMatrix(1, OUT_DIM, HIDDEN_DIM, fout, this->weights2, out2);

	//�������� ������������ �������� �� �������� (out2) �� ���������� (bias2), ������ � out2
	sumMatrix(1, OUT_DIM, out2, this->bias2, out2);

	//���������� ������� ��������� ��� ������� ������� OUT, �������� ����� ������ fout
	//������ ���� �������� �������� � ��� ������ ������� �������������, ������� ���������� ������� softmax, ������� out2, �������� fout2
	double* fout2 = this->softmax(out2, OUT_DIM);

	//����������� ������
	delete[] out;
	delete[] fout;
	delete[] out2;
	//���������� ��������� �� ������ � �������������� ������������
	return fout2;
}

double* Network::to_full(int value)
{
	/*
	* ������ ����� ��������� ������ ����[0, 0, ..., 0, 1, 0], �������� OUT_DIM, ��� value - ������ ����� 1
	* ��������� ��� ���������� ������ ��� �������� ���������
	* �� ���� ����������� ���������� ����� �� ��������� ���� � ������ ���������� ������������� �������������, ������� ������ ������ ���������
	*/

	//��������� ������ ��� �������� ��������, ����������� OUT_DIM (10)
	double* result = new double[OUT_DIM];

	//���������� ������� ������
	for (int i = 0; i < OUT_DIM; i++)
	{
		result[i] = 0.0;
	}

	//������� �� ������� value ����� ����� �������� 1
	result[value] = 1.0;

	//����������� ��������� �� �������� ������
	return result;
}

void Network::loadDataset(DataSetType type)
{
	/*
	��������� ������� � ������ ������ �� ������
	� ����������� �� ���� ��������
	(
		DataSetType::Train - ������������� �������, 60� ��������
		DataSetType::Test - ����������� �������, 10� ��������
	)
	��� �������������� �������� ��������� ���� train-images.idx3-ubyte,
	��� ��������� �������� �������� ��������� ���� t10k-images.idx3-ubyte
	����� ������ ���� � ���������� �������
	*/
	switch (type)
	{
	case Train:
		//������������� �������
		this->datasetSize = 60000;
		break;
	case Test:
		//�������� �������
		this->datasetSize = 10000;
		break;
	}
	//�������� ������� �� datasetSize ���������, �� 784 ������� � ������ ������
	double** data = readMNIST(type);
	if (data != nullptr)
	{
		this->dataset = data;
	}
	else
	{
		cout << "�������� � ������� � ��������" << endl;
	}
	/*
	* �������� ������ ������� � ��������, �.�. ������� ������� �� ������� data ������������� ������� �� ������ data2,
	* ������: �� �������� ���������� ����� 3
	* data[0] = [0, 1, 0 .... (784 �������)]
	* data2[0] = 3
	*/
	int* data2 = readMNISTLabels(type);
	if (data != nullptr)
	{
		this->dataAnswers = data2;
	}
	else
	{
		cout << "�������� � ������� �������� � ��������" << endl;
	}

	//������� �������������� ���������
	string name = type == DataSetType::Test ? "��������" : "�������������";
	cout << "�������� " << name << " �������" << endl;
}

void Network::training()
{
	/*
	* ����� �������� ���������
	* ��� ������ ����� ����� ��������� datasetSize ������������ � �������������
	* ������������ �������� �������� - ����������� �����
	* ������������ �������� ���������� ��������� - �������� ���������������
	* 
	* ����������� ���� ��������� ����
	* �������, �������, ������� � �������� ���������, ��������, �������� � �������� ���������
	* trainX      t1       h1                            t2        h2
	*/
	cout << "���������� ������" << endl;
	deleteMatrix(this->losses);
	//�������������� ������ ������, ���������� � ���� ��������
	this->losses = new double[this->datasetSize * EPOCH_COUNT];

	//��������� ���������� ���� ���
	for (int epoch = 0; epoch < EPOCH_COUNT; epoch++)
	{
		cout << epoch << " �����" << endl;
		//��� ������� �������� �� �������� �������� (���� ����������� ������� � �� ��������, �� �����)
		for (int i = 0; i < this->datasetSize; i++)
		{
			//�������� ������� ������� ������, ������ �� 784 �������� � ��������� [0, 1]
			double* trainX = this->dataset[i];
			//�������� ����� ��� ������� ������, �����
			int trainY = this->dataAnswers[i];

			//������ ���, ��������� � ������ predict
			double* out = new double[HIDDEN_DIM];
			multiplyMatrix(1, HIDDEN_DIM, INPUT_DIM, trainX, this->weights1, out);
			sumMatrix(1, HIDDEN_DIM, out, this->bias1, out);

			double* fout = this->relu(out, HIDDEN_DIM);

			double* out2 = new double[OUT_DIM];
			multiplyMatrix(1, OUT_DIM, HIDDEN_DIM, fout, this->weights2, out2);
			sumMatrix(1, OUT_DIM, out2, this->bias2, out2);

			double* z = this->softmax(out2, OUT_DIM);
			//����� ������� ����

			//�������� ���
			//���������� ������
			double E = this->sparse_cross_entropy(z, trainY);

			//�������������� ������ � ���� �������� �������� ��������� ����
			double* y_full = this->to_full(trainY);

			//������� ����������� ������ E �� ������� ������� �������� �������� ��������� ����
			double* dE_dt2 = new double[OUT_DIM];
			diffMatrix(1, OUT_DIM, z, y_full, dE_dt2);

			//������� ����������� ������ E �� ������� W2
			double* dE_dW2 = new double[HIDDEN_DIM * OUT_DIM];

			//������������� ������� fout, �������� fout_transposed
			double* fout_transposed = transposeMatrix(1, HIDDEN_DIM, fout);

			//����������� ������� fout_transposed �� de_dt2, ���������� � dE_dW2
			multiplyMatrix(HIDDEN_DIM, OUT_DIM, 1, fout_transposed, dE_dt2, dE_dW2);

			//������� ����������� ������ E �� ������� h1
			double* dE_dh1 = new double[HIDDEN_DIM];

			//������������� ������� weights2, �������� weights2_transposed
			double* weights2_transposed = transposeMatrix(HIDDEN_DIM, OUT_DIM, this->weights2);

			//����������� ������� dE_dt2 �� weights2_transposed, ���������� � dE_dh1
			multiplyMatrix(1, HIDDEN_DIM, OUT_DIM, dE_dt2, weights2_transposed, dE_dh1);

			//������� ����������� ������ E �� ������� t1
			double* dE_dt1 = new double[HIDDEN_DIM];

			//���������� dD_dt1
			for (int k = 0; k < HIDDEN_DIM; k++)
			{
				//������ ������� ����� ����������� �� ������� ��������� ��� �������� ����, ����������� �� ������� ����������� ������ E �� ������� h1
				dE_dt1[k] = this->drelu(out[k]) * dE_dh1[k];
			}

			//������� ����������� ������ E �� ������� W1
			double* dE_dW1 = new double[INPUT_DIM * HIDDEN_DIM];

			//������������� ������� trainX, �������� trainX_transposed
			double* trainX_transposed = transposeMatrix(1, INPUT_DIM, trainX);
			
			//����������� ������� trainX_transposed �� dE_dt1, ���������� � dE_dW1
			multiplyMatrix(INPUT_DIM, HIDDEN_DIM, 1, trainX_transposed, dE_dt1, dE_dW1);

			//�������� ���������� �������� ��������� �� ��� �������� ALPHA � ������������ ���� �� ���������� ��������
			//��� ����� � �������� �� ������� ����
			multiplyMatrixNumber(dE_dW1, INPUT_DIM * HIDDEN_DIM, ALPHA);
			diffMatrix(INPUT_DIM, HIDDEN_DIM, this->weights1, dE_dW1, this->weights1);
			
			//��� �������� � �������� �� ������� ����
			multiplyMatrixNumber(dE_dt1, HIDDEN_DIM, ALPHA);
			diffMatrix(1, HIDDEN_DIM, this->bias1, dE_dt1, this->bias1);

			//��� ����� �� �������� ���� �� �������� ����
			multiplyMatrixNumber(dE_dW2, HIDDEN_DIM * OUT_DIM, ALPHA);
			diffMatrix(HIDDEN_DIM, OUT_DIM, this->weights2, dE_dW2, this->weights2);

			//��� �������� �� �������� ���� �� �������� ����
			multiplyMatrixNumber(dE_dt2, OUT_DIM, ALPHA);
			diffMatrix(OUT_DIM, 1, this->bias2, dE_dt2, this->bias2);
			
			//���������� �������� ������ � ������ ������
			//� ������ ��������� ������ ������ ����� �� ������������, ������������ �� ���� ����� ������� ������ ������ ��� ������������ ������������ � ������ ����������
			this->losses[i + epoch * this->datasetSize] = E;

			//����������� ������
			delete[] out;
			delete[] fout;
			delete[] fout_transposed;
			delete[] out2;
			delete[] z;
			delete[] y_full;
			delete[] dE_dt2;
			delete[] dE_dW2;
			delete[] dE_dh1;
			delete[] weights2_transposed;
			delete[] dE_dt1;
			delete[] dE_dW1;
			delete[] trainX_transposed;
		}
	}
}

void Network::calcAccuracy()
{
	/*
	* ����� ������������ ������� ������ � �������� ������ � ������� �������� � ������� ��� �� �����
	*/
	int count = 0;
	//��� ������� �������� ��������
	for (int i = 0; i < this->datasetSize; i++)
	{
		//�������� ������� ��������
		double* test_data = this->dataset[i];
		//���������� �����
		int correct = this->dataAnswers[i];
		//�������� ������������ �� ���������
		double* z = this->predict(test_data);
		//������������� �� �������������� ������������� � �����
		int y_predict = argMax(z, 10);
		//���� ����������� �����
		if (y_predict == correct)
		{
			//��������� ���������� ����� ������������� �����
			count += 1;
		}
	}
	//������� ��������� ��������� ������������� � ������� ��������
	double accuracy = double(count) / double(datasetSize);
	cout << "��������: " << accuracy << endl;
}

void Network::saveWeights()
{
	/*
	* ����� ���������� ������� ����� � �������� � ����� w1, w2 � b1, b2 ��������������
	*/
	ofstream fout;
	fout.open("w1");
	if (!fout.is_open()) {
		cout << "������, �� ������� ������� ���� ��� ���������� �����" << endl;
	}

	for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++)
	{
		fout << this->weights1[i];
	}
	fout.close();
	fout.open("b1");
	for (int i = 0; i < HIDDEN_DIM; i++)
	{
		fout << this->bias1[i];
	}
	fout.close();
	fout.open("w2");
	for (int i = 0; i < HIDDEN_DIM * OUT_DIM; i++)
	{
		fout << this->weights2[i];
	}
	fout.close();
	fout.open("b2");
	for (int i = 0; i < OUT_DIM; i++)
	{
		fout << this->bias2[i];
	}
	fout.close();
	cout << "Weights saved \n";
}

void Network::loadWeights()
{
	/*
	* ����� �������� ����� � �������� �� ������ w1, w2 � b1, b2 ��������������
	*/
	ifstream file("w1");
	if (file.is_open())
	{
		int count = INPUT_DIM * HIDDEN_DIM;
		double* data = new double[count];

		for (int i = 0; i < count; i++)
		{
			file >> data[i];
		}
		delete[] this->weights1;
		this->weights1 = data;
		cout << "����1 ���������" << endl;
	}
	file.close();

	file.open("b1");
	if (file.is_open())
	{
		int count = INPUT_DIM;
		double* data2 = new double[count];

		for (int i = 0; i < count; i++)
		{
			file >> data2[i];
		}
		delete[] this->bias1;
		this->bias1 = data2;
		cout << "��������1 ���������" << endl;
	}
	file.close();

	file.open("w2");
	if (file.is_open())
	{
		int count = HIDDEN_DIM * OUT_DIM;
		double* data3 = new double[count];

		for (int i = 0; i < count; i++)
		{
			file >> data3[i];
		}
		delete[] this->weights2;
		this->weights2 = data3;
		cout << "����2 ���������" << endl;
	}
	file.close();

	file.open("b2");
	if (file.is_open())
	{
		int count = OUT_DIM;
		double* data4 = new double[count];

		for (int i = 0; i < count; i++)
		{
			file >> data4[i];
		}
		delete[] this->bias2;
		this->bias2 = data4;
		cout << "��������2 ���������" << endl;
	}
	file.close();
}

void Network::showLayers()
{
	//���������� ����� ��� ����������� �������������� ������ � ��������
	statsArray(INPUT_DIM * HIDDEN_DIM, this->weights1);
	statsArray(HIDDEN_DIM, this->bias1);
	statsArray(HIDDEN_DIM * OUT_DIM, this->weights2);
	statsArray(OUT_DIM, this->bias2);
}