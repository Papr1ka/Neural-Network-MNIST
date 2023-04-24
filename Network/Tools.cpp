#include "Tools.h"


void randomWeights(double* array, int n)
{
    //Инициализирует матрицу рандомными значениями от 0 до 1
	for (int i = 0; i < n; i++)
	{
        /*
        * Внимание, для задачи классификации цифр, а конкретно для функции softmax обязательно добавить множитель, сокращающий диапазон весов
        * Здесь сокращено до отрезка[0, 0.1] при помощи множителя 0.1
        * Дело в том, что при некоторых сценариях в функции softmax может возникнуть переполнение
        */
        array[i] = (double)(rand()) / RAND_MAX * 0.1;
	}
}

double* initMatrix(int n, int m)
{
    //Создаёт матрицу (nxm) и заполняет её числами от 0 до 1, возвращает указатель
    //Создаёт
    double* array = new double[n * m];
    //Заполняет
    randomWeights(array, n * m);
    //Возвращает указатель на получившуюся матрицу
    return array;
}

void multiplyMatrix(int M, int N, int K, const double* A, const double* B, double* C)
{
    //Произведение матриц A[MxK], B[KxN], запись в матрицу C[MxN]
    //Матрицы представленны в одномерном виде, где строки идут друг за другом
    for (int i = 0; i < M; ++i)
    {
        double* c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            const double* b = B + k * N;
            double a = A[i * K + k];
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
}

void multiplyMatrixNumber(double* Matrix, int size, double number)
{
    //Произведение матрицы на число, запись производится в переданную матрицу
    for (int i = 0; i < size; i++)
    {
        Matrix[i] = number * Matrix[i];
    }
}

void deleteMatrix(double* array)
{
    //Удаляет матрицу
    if (array != nullptr)
    {
        delete[] array;
    }
}

void sumMatrix(int M, int N, const double* A, const double* B, double* C)
{
    //Сумма матриц, запись производится в матрицу C
    int right = M * N;
    for (int i = 0; i < right ; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void diffMatrix(int M, int N, const double* A, const double* B, double* C)
{
    //Разность матриц, запись производится в матрицу C
    int right = M * N;
    for (int i = 0; i < right; i++)
    {
        C[i] = A[i] - B[i];
    }
}

using namespace std;
int ReverseInt(int i)
{
    //Вспомогательный метод для чтения датасета
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

double** readMNIST(DataSetType type)
{
    //Чтение датасета Train - тренировочный, Test - тестирования, необходимо наличие соответствующих файлов в рабочей директории
    int DataOfAnImage = 28 * 28;
    int NumberOfImages;
    string fileName;

    switch (type)
    {
    case Train:
        NumberOfImages = 60000;
        fileName = "train-images.idx3-ubyte";
        break;
    case Test:
        NumberOfImages = 10000;
        fileName = "t10k-images.idx3-ubyte";
        break;
    default:
        NumberOfImages = 60000;
        fileName = "train-images.idx3-ubyte";
        break;
    }
    double** data = new double*[NumberOfImages];

    for (int i = 0; i < NumberOfImages; i++)
    {
        data[i] = new double[DataOfAnImage];
    }

    ifstream file(fileName, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for (int i = 0; i < number_of_images; ++i)
        {
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    data[i][(n_rows * r) + c] = (double)temp / 255;
                }
            }
        }
        return data;
    }
    else
    {
        cout << "Ошибка, файл {" << fileName << "} не найден";
        return nullptr;
    }
}

int* readMNISTLabels(DataSetType type)
{
    //Чтение ответов к датасету Train - тренировочный, Test - тестирования, необходимо наличие соответствующих файлов в рабочей директории
    int NumberOfImages;
    string fileName;

    switch (type)
    {
    case Train:
        NumberOfImages = 60000;
        fileName = "train-labels.idx1-ubyte";
        break;
    case Test:
        NumberOfImages = 10000;
        fileName = "t10k-labels.idx1-ubyte";
        break;
    default:
        NumberOfImages = 60000;
        fileName = "train-labels.idx1-ubyte";
        break;
    }
    int* data = new int [NumberOfImages];

    ifstream file(fileName, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_items = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*)&number_of_items, sizeof(number_of_items));
        for (int i = 0; i < NumberOfImages; ++i)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            data[i] = (int)temp;
        }
        return data;
    }
    else
    {
        cout << "Ошибка, файл {" << fileName << "} не найден";
        return nullptr;
    }
}

int argMax(double* value, int size)
{
    //Возвращает индекс наибольшего элемента в массиве
    double max = value[0]; //текущий максимум
    int prediction = 0; //индекс наибольшего элемента
    double tmp; //временная переменная
    for (int j = 1; j < size; j++)
    {
        tmp = value[j];
        if (tmp > max) {
            //если элемент больше текущего, меняем больший на текущий
            prediction = j;
            max = tmp;
        }
    }
    //Возвращаем индекс
    return prediction;
}

double* transposeMatrix(int M, int N, double* Matrix)
{
    //Транспонирование матрицы M на N, возвращает новую матрицу, старую не изменяет
    //Матрицы представленны в одномерном виде, где строки идут друг за другом
    //Возвращает новый массив
    double* result = new double[M * N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            result[i * M + j] = Matrix[j * N + i];
        }
    }
    return result;
}

void statsArray(int n, double* array)
{
    //Выводит статистику по массиву: размер, минимум, максимум, среднее
    double min = array[0];
    double max = array[0];
    double avg = 0;

    for (int i = 0; i < n; i++)
    {
        if (array[i] > max)
        {
            max = array[i];
        }
        if (array[i] < min)
        {
            min = array[i];
        }
        avg += array[i];
    }
    avg = avg / double(n);
    cout << "Массив [" << n << "], Min: " << min << " Max: " << max << " Avg: " << avg << endl;
}

double* readFile(string filename)
{
    //Метод предназначен для чтения файла image по адресу filename
    //Файл должен представлять из себя столбец значений пикселей в диапазоне [0, 1]
    //Возвращает указатель на созданный массив, размером 784
    ifstream file(filename);
    if (file.is_open())
    {
        int count = 784;
        double* data = new double[count];

        for (int i = 0; i < count; i++)
        {
            file >> data[i];

        }
        cout << "Файл загружен" << endl;
        file.close();
        return data;
    }
    else
    {
        cout << "Файл не найден" << endl;
    }
    return nullptr;
}
