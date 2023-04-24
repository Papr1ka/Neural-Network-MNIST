#pragma once
#include <fstream> //Библиотека для работа с файлами
#include <iostream> //Библиотека для работы с потоком вывода в консоль
#include <string> //Библиотека для работы со строками

using namespace std;

enum DataSetType
{
	Train, //Тренеровочный датасет
	Test //Тестовый датасет
};

void randomWeights(double* array, int n); //Инициализирует матрицу рандомными значениями от 0 до 1

double* initMatrix(int n, int m); //Создаёт матрицу (nxm) и заполняет её числами от 0 до 1, возвращает указатель

void multiplyMatrix(int M, int N, int K, const double* A, const double* B, double* C); //Произведение матриц A[MxK], B[KxN], запись в матрицу C[MxN]

void multiplyMatrixNumber(double* Matrix, int size, double number); //Произведение матрицы на число, запись производится в переданную матрицу

void deleteMatrix(double* array); //Удаляет матрицу

void sumMatrix(int M, int N, const double* A, const double* B, double* C); //Сумма матриц, запись производится в матрицу C

void diffMatrix(int M, int N, const double* A, const double* B, double* C); //Разность матриц, запись производится в матрицу C

int ReverseInt(int i); //Вспомогательный метод для чтения датасета

double** readMNIST(DataSetType type); //Чтение датасета Train - тренировочный, Test - тестирования, необходимо наличие соответствующих файлов в рабочей директории

int* readMNISTLabels(DataSetType type); //Чтение ответов к датасету Train - тренировочный, Test - тестирования, необходимо наличие соответствующих файлов в рабочей директории

int argMax(double* value, int size); //Возвращает индекс наибольшего элемента в массиве

double* transposeMatrix(int M, int N, double* Matrix); //Транспонирование матрицы M на N, возвращает новую матрицу, старую не изменяе

void statsArray(int n, double* array); //Выводит статистику по массиву: размер, минимум, максимум, среднее

double* readFile(string filename); //Метод предназначен для чтения файла image по адресу filename
