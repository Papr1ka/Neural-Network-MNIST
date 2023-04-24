#pragma once
#include <fstream> //���������� ��� ������ � �������
#include <iostream> //���������� ��� ������ � ������� ������ � �������
#include <string> //���������� ��� ������ �� ��������

using namespace std;

enum DataSetType
{
	Train, //������������� �������
	Test //�������� �������
};

void randomWeights(double* array, int n); //�������������� ������� ���������� ���������� �� 0 �� 1

double* initMatrix(int n, int m); //������ ������� (nxm) � ��������� � ������� �� 0 �� 1, ���������� ���������

void multiplyMatrix(int M, int N, int K, const double* A, const double* B, double* C); //������������ ������ A[MxK], B[KxN], ������ � ������� C[MxN]

void multiplyMatrixNumber(double* Matrix, int size, double number); //������������ ������� �� �����, ������ ������������ � ���������� �������

void deleteMatrix(double* array); //������� �������

void sumMatrix(int M, int N, const double* A, const double* B, double* C); //����� ������, ������ ������������ � ������� C

void diffMatrix(int M, int N, const double* A, const double* B, double* C); //�������� ������, ������ ������������ � ������� C

int ReverseInt(int i); //��������������� ����� ��� ������ ��������

double** readMNIST(DataSetType type); //������ �������� Train - �������������, Test - ������������, ���������� ������� ��������������� ������ � ������� ����������

int* readMNISTLabels(DataSetType type); //������ ������� � �������� Train - �������������, Test - ������������, ���������� ������� ��������������� ������ � ������� ����������

int argMax(double* value, int size); //���������� ������ ����������� �������� � �������

double* transposeMatrix(int M, int N, double* Matrix); //���������������� ������� M �� N, ���������� ����� �������, ������ �� �������

void statsArray(int n, double* array); //������� ���������� �� �������: ������, �������, ��������, �������

double* readFile(string filename); //����� ������������ ��� ������ ����� image �� ������ filename
