﻿#include <iostream> //Библиотека для работы с потоком вывода на экран
#include "Network.h" //Файл с нашей нейросетью

using namespace std;

void trainOnWeights()
{
    /*
    * Процедура производит обучений нейросети с использованием записанных в файлы w1, w2, b1, b2 весов и смещений
    * Подсчитывает получившуюся точность на тестовом датасете
    * Сохраняет получившиеся веса и смещения в файлы
    */

    //Инициализация нейросети
    Network* net = new Network();
    //Загрузка имеющихся весов и смещений из соответствующих файлов
    net->loadWeights();

    //Загрузка тестового датасета
    net->loadDataset(DataSetType::Test);
    //Подсчёт точности до обучения
    net->calcAccuracy();

    //Загрузка тренеровочного датасета
    net->loadDataset(DataSetType::Train);
    //Обучение на текущих весах и смещениях
    net->training();

    //Загрузка тестового датасета
    net->loadDataset(DataSetType::Test);
    //подсчёт точности после тренировки
    net->calcAccuracy();
    //Сохранение весов и смещений в соответствующие файлы
    net->saveWeights();
    //Удаление нейросети
    delete net;
}

void train()
{
    /*
    * Процедура производит обучений нейросети с нуля на тренеровочном датасете
    * Подсчитывает получившуюся точность на тестовом датасете
    * Сохраняет получившиеся веса и смещения в файлы
    */

    //Инициализация нейросети
    Network* net = new Network();

    //Загрузка тренеровочного датасета
    net->loadDataset(DataSetType::Train);
    //Обучение нейросети на текущем датасете
    net->training();

    //Загрузка тестового датасете
    net->loadDataset(DataSetType::Test);
    //Подсчёт точности после тренировки
    net->calcAccuracy();
    //Сохранение получившихся весов в файл
    net->saveWeights();
    //Удаление нейросети
    delete net;
}

void predictTest()
{
    /*
    * Процедура производит предсказание о принадлежности изображению к соответствующему классу (цифра 1, ..., цифра 9)
    * Открывает файл image, в котором должна находиться закодированная фотография цифры,
    * цифру следует закодировать на стороне в 1-м канале (чёрно-белый) и разрешением 28 на 28 пикселей,
    * для удобства в директории проекта написан скрипт на Python - convert.py, требует python3 + Pillow + NumPy,
    * файл для кодирования следует назвать test.png
    */

    //Инициализация нейросети
    Network* net = new Network();
    //Загружаем веса и смещения из файлов, если они есть
    net->loadWeights();
    
    //Чтение файла image, получаем массив из 784 пикселей со значениями в интервале [0, 1]
    double* array = readFile("image");
    //Выводим закодированное изображение в консоль для удобства отладки
    if (array != nullptr)
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                cout << array[i * 28 + j] << " ";
            }
            cout << endl;
        }
    }

    /*
    * Предсказываем принадлежность к классу на основе массива пикселей,
    * получаем вероятностное распределение из 10-ти элементов (количества нейронов выходного слоя элементов)
    * Каждый элемент соответствует вероятности принодлежности к классу (цифра 0, ..., цифра 9)
    * Предсказание делается на основе текущих весов
    */
    double* test = net->predict(array);

    //Выводим вероятностное распределение в консоль
    for (int i = 0; i < OUT_DIM; i++)
    {
        cout << test[i] << " ";
    }
    cout << endl;

    //Число будет равно индексу элемента с максимальной вероятностью
    cout << "Предсказание, это число: " << argMax(test, 10) << endl;

    //Удаляем нейросеть
    delete net;
}

void checkAccuracy()
{
    //Процедура подсчитывает и выводит в консоль текущую точность нейросети, определение точности производится на тестовом датасете

    //Инициализация нейросети
    Network* net = new Network();
    //Загрузка текущих весов из файлов, если файлов нет, будут использованы случайные числа
    net->loadWeights();
    //Загрузка тестового датасета
    net->loadDataset(DataSetType::Test);
    //Подсчёт точности нейросети на текущем (тестовом) датасете
    net->calcAccuracy();
    //Удаляем нейросеть
    delete net;
}


int main(int argc, char* argv[])
{
    //Устанавливаем русскую локализацию консоли для вывода информационных сообщений на русском языке
    setlocale(LC_ALL, "Rus");

    //Вывод количеств нейронов на каждом слое
    cout << INPUT_DIM << endl; //Входной слой
    cout << HIDDEN_DIM << endl; //Скрытый слой
    cout << OUT_DIM << endl; //Выходной слой
    
    //Выполним процедуру обучения нейросети
    //train();
    predictTest();
    //И предскажем цифру, изображённую в файле test.png
    /*
    * Для удобства написаны следующие процедуры :
    * 
    * checkAccuracy - Подсчитывает и выводит в консоль текущую точность нейросети на тестовом датасете
    * 
    * predictTest - Предсказывает на основе закодированного изображения image в директории проекта
    * 
    * trainOnWeights - Запускает обучение нейросети, используя сохранённые ранее веса и смещений
    * 
    * train - Запускает обучение нейросети, генерируя веса и смещения случайным образом
    */

    return 0;
}