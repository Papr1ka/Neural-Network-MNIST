#include "Network.h"

Network::Network()
{
	//Конструктор нейросети, формирует массивы сдвигов и смещений, заполненные случайными числами
	this->weights1 = initMatrix(INPUT_DIM, HIDDEN_DIM);
	this->bias1 = initMatrix(1, HIDDEN_DIM);
	this->weights2 = initMatrix(HIDDEN_DIM, OUT_DIM);
	this->bias2 = initMatrix(1, OUT_DIM);
}

Network::~Network()
{
	//Деструктор, освобождает память, удаляя все созданные массивы
	deleteMatrix(this->weights1);
	deleteMatrix(this->bias1);
	deleteMatrix(this->weights2);
	deleteMatrix(this->bias2);
	deleteMatrix(this->losses);
}

double* Network::relu(double* array, int n)
{
	/*
	* Функция активации relu, описывается как (x, если x > 0, иначе 0), имеет производную, что важно для обучения
	* Функцию можно заменить на любую другую, например сигмоид, тогда следует переписать и производную от этой функции (метод drelu)
	* Функция активации принимает выходной сигнал из предыдущей ячейки и преобразует его в некоторую форму,
	* которую можно использовать в качестве входных данных для следующей ячейки.
	* Функция активации добавляет нелинейность в нейронную сеть, что позволяет ей делать более сложные "Выводы"
	*/
	
	//Создание выходного массива для будущих выходных значений
	double* outArray = new double[n];
	//Применение функции активации к каждому элементу входного массива и запись в выходной массив
	for (int i = 0; i < n; i++)
	{
		outArray[i] = array[i] >= 0 ? array[i] : 0;
	}
	//Возвращение указателя на выходной массив
	return outArray;
}

double* Network::softmax(double* array, int n)
{
	/*
	* Функция активации softmax, описывается как (ei^x / sum(ej^x), где j меняется от 0 до n, а i это индекс элемента массива), 
	* Для каждого элемента рассчитывается экспонента в степени значения элемента, делённое на сумму экспонент в степени значений элементов всего входного массива,
	* Софтмакс – функция, превращающая логиты (наборы чисел) в вероятности, причем сумма последних равна единице.
	* Функция выводит в качестве результата вектор, представляющий распределения вероятностей списка потенциальных результатов.
	* Данная функция отлично подходит для задачи классификации
	*/

	//Создание выходного массива для будущих выходных значений
	double* outArray = new double[n];
	
	//Сумма всех экспонент в степени элементов
	double summa = 0;

	//Проход по всему массиву
	for (int i = 0; i < n; i++)
	{
		//Вычисление експоненты в степени элемента
		double item = exp(array[i]);
		//Запись в выходной массив
		outArray[i] = item;
		//Прибавление к сумме
		summa += item;
	}

	//Проход по всему массиву
	for (int i = 0; i < n; i++)
	{
		//Вычисление итогового значения для каждого элемента, экспонета в степени элемента, делённая на сумму таких экспонент
		outArray[i] = outArray[i] / summa;
	}

	//Возвращение указателя на выходной массив
	return outArray;
}


double Network::drelu(double x)
{
	/*
	* Дифференциал от функции активации relu, описывается как (1, если x > 0, иначе 0)
	* Необходима для обучения нейросети
	*/
	return x >= 0 ? 1. : 0.;
}

double Network::sparse_cross_entropy(double* array, int value)
{
	/*
	* 
	* Функция вычисления ошибки, категориальная-кросс-энтропия
	* Используется для обучения нейросети
	* Функция подсчёта ошибки может быть разной, но для задачи классификации лучше всего подходим именно эта
	* Функция записана в укороченном виде, где оставлена только значимая часть, следует получить больше информации в интернете
	*/
	return -log(array[value]);
}

double* Network::predict(double* data)
{
	/*
	* Метод предсказания, прямой ход
	* Преобразования ведутся следующим образом, пусть у нас есть 2 входных нейрона (N1, N2) и 2 выходных (T1, T2), тогда
	* нам понядобятся веса (W11 от N1 к T1, W12 от N1 к T2, W21 от N2 к T1, W22 от N2 к T2), а также смещения (bias1 для T1 и bias2 для T2)
	* Входные значения на выходных нейронах будут иметь вид:
	* T1 = N1*w11 + N2*w21 + bias1
	* T2 = N1*w12 + N2*w22 + bias2
	* Данные операция можно представить в следующем виде:
	* Пусть у нас есть матрица весов
	* W (2x2) =
	* (w11, w12,
	*  w21, w22)
	* Матрица входных нейронов (1x2) N = (N1, N2)
	* Матрица смещений (1x2) B = (bias1, bias2)
	* Тогда матрицу выходных нейронов (1x2) OUT можно получить так:
	* OUT = N * W + B
	* Следующим этапом вычисление функции активации для каждого из элементов OUT
	* Это была процедура вычислений следующего слоя нейронов на основе предыдущего, следующие слои будут использовать OUT как входные
	*/
	//Выделение памяти под итоговые значения на скрытом слое нейронов
	double* out = new double[HIDDEN_DIM];

	//Перемножение матриц (data), размерности (1x784) на матрицу весов (weights1), размерности (784x128) и запись результата в массив out
	multiplyMatrix(1, HIDDEN_DIM, INPUT_DIM, data, this->weights1, out);
	
	//Сложение получившихся значений на нейронах (out) со смещениями (bias1), запись в out
	sumMatrix(1, HIDDEN_DIM, out, this->bias1, out);

	//Вычисление функции активации для каждого нейрона OUT, получаем новый массив fout
	double* fout = this->relu(out, HIDDEN_DIM);



	//Выделение памяти под итоговые значения на выходном слое нейронов
	double* out2 = new double[OUT_DIM];

	//Перемножение матриц (fout), размерности (1x128) на матрицу весов (weights2), размерности (128x10) и запись результата в массив out2
	multiplyMatrix(1, OUT_DIM, HIDDEN_DIM, fout, this->weights2, out2);

	//Сложение получившихся значений на нейронах (out2) со смещениями (bias2), запись в out2
	sumMatrix(1, OUT_DIM, out2, this->bias2, out2);

	//Вычисление функции активации для каждого нейрона OUT, получаем новый массив fout
	//Данный слой является выходным и его выдаче готовых вероятностней, поэтому используем функцию softmax, передаём out2, получаем fout2
	double* fout2 = this->softmax(out2, OUT_DIM);

	//Освобождаем память
	delete[] out;
	delete[] fout;
	delete[] out2;
	//Возвращаем указатель на массив с распределением вероятностей
	return fout2;
}

double* Network::to_full(int value)
{
	/*
	* Данный метод формирует массив вида[0, 0, ..., 0, 1, 0], размером OUT_DIM, где value - индекс числа 1
	* Необходим для вычисления ошибок при обучении нейросети
	* По сути преобразует правильный ответ из числового вида в массив правильных вероятностных распределений, которые должна выдать нейросеть
	*/

	//Выделение памяти под выходные значения, размерность OUT_DIM (10)
	double* result = new double[OUT_DIM];

	//Заполнение массива нулями
	for (int i = 0; i < OUT_DIM; i++)
	{
		result[i] = 0.0;
	}

	//Элемент по индексу value будет иметь значение 1
	result[value] = 1.0;

	//Возвращение указателя на выходной массив
	return result;
}

void Network::loadDataset(DataSetType type)
{
	/*
	Загружает датасет в объект класса из файлов
	В зависимости от типа датасета
	(
		DataSetType::Train - тренеровочный датасет, 60к картинок
		DataSetType::Test - проверочный датасет, 10к картинок
	)
	Для тренеровочного датасета необходим файл train-images.idx3-ubyte,
	Для тестового датасета датасета необходим файл t10k-images.idx3-ubyte
	Файлы должны быть в директории проекта
	*/
	switch (type)
	{
	case Train:
		//Тренеровочный датасет
		this->datasetSize = 60000;
		break;
	case Test:
		//Тестовый датасет
		this->datasetSize = 10000;
		break;
	}
	//Получаем матрицу из datasetSize элементов, по 784 пикселя в каждой записи
	double** data = readMNIST(type);
	if (data != nullptr)
	{
		this->dataset = data;
	}
	else
	{
		cout << "Проблема с данными в датасете" << endl;
	}
	/*
	* Получаем массив ответов к датасету, т.е. каждому массиву из матрицы data соответствует элемент из массив data2,
	* Пример: на картинке изображено число 3
	* data[0] = [0, 1, 0 .... (784 пискеля)]
	* data2[0] = 3
	*/
	int* data2 = readMNISTLabels(type);
	if (data != nullptr)
	{
		this->dataAnswers = data2;
	}
	else
	{
		cout << "Проблема с данными надписей к датасету" << endl;
	}

	//Выводим информационное сообщение
	string name = type == DataSetType::Test ? "Тестовый" : "Тренировочный";
	cout << "Загружен " << name << " Датасет" << endl;
}

void Network::training()
{
	/*
	* Метод обучения нейросети
	* Для каждой эпохи будет выполнено datasetSize предсказаний и корректировок
	* Используемый алгоритм обучения - Градиентный спуск
	* Используемый алгоритм вычисления градиента - обратное распространение
	* 
	* Наименовани слоёв нейронной сети
	* входной, скрытый, скрытый с функцией активации, выходной, выходной с функцией активации
	* trainX      t1       h1                            t2        h2
	*/
	cout << "Тренировка начата" << endl;
	deleteMatrix(this->losses);
	//Инициализируем массив ошибок, полученных в ходе обучения
	this->losses = new double[this->datasetSize * EPOCH_COUNT];

	//Повторять количество эпох раз
	for (int epoch = 0; epoch < EPOCH_COUNT; epoch++)
	{
		cout << epoch << " Эпоха" << endl;
		//Для каждого элемента из текущего датасета (есть возможность обучать и на тестовом, но нужды)
		for (int i = 0; i < this->datasetSize; i++)
		{
			//Получаем текущие входные данные, массив из 784 пикселей в диапазоне [0, 1]
			double* trainX = this->dataset[i];
			//Получаем ответ для входных данных, число
			int trainY = this->dataAnswers[i];

			//Прямой ход, описанный в методе predict
			double* out = new double[HIDDEN_DIM];
			multiplyMatrix(1, HIDDEN_DIM, INPUT_DIM, trainX, this->weights1, out);
			sumMatrix(1, HIDDEN_DIM, out, this->bias1, out);

			double* fout = this->relu(out, HIDDEN_DIM);

			double* out2 = new double[OUT_DIM];
			multiplyMatrix(1, OUT_DIM, HIDDEN_DIM, fout, this->weights2, out2);
			sumMatrix(1, OUT_DIM, out2, this->bias2, out2);

			double* z = this->softmax(out2, OUT_DIM);
			//Конец прямого хода

			//Обратный ход
			//Вычисление ошибки
			double E = this->sparse_cross_entropy(z, trainY);

			//Преобразование ответа к виду выходных значений нейронной сети
			double* y_full = this->to_full(trainY);

			//Частная производная ошибки E по матрице массиву выходных значений нейронной сети
			double* dE_dt2 = new double[OUT_DIM];
			diffMatrix(1, OUT_DIM, z, y_full, dE_dt2);

			//Частная производная ошибки E по матрице W2
			double* dE_dW2 = new double[HIDDEN_DIM * OUT_DIM];

			//Транспонируем матрицу fout, получаем fout_transposed
			double* fout_transposed = transposeMatrix(1, HIDDEN_DIM, fout);

			//Перемножаем матрицы fout_transposed на de_dt2, записываем в dE_dW2
			multiplyMatrix(HIDDEN_DIM, OUT_DIM, 1, fout_transposed, dE_dt2, dE_dW2);

			//Частная производная ошибки E по массиву h1
			double* dE_dh1 = new double[HIDDEN_DIM];

			//Транспонируем матрицу weights2, получаем weights2_transposed
			double* weights2_transposed = transposeMatrix(HIDDEN_DIM, OUT_DIM, this->weights2);

			//Перемножаем матрицы dE_dt2 на weights2_transposed, записываем в dE_dh1
			multiplyMatrix(1, HIDDEN_DIM, OUT_DIM, dE_dt2, weights2_transposed, dE_dh1);

			//Частная производная ошибки E по массиву t1
			double* dE_dt1 = new double[HIDDEN_DIM];

			//Вычисление dD_dt1
			for (int k = 0; k < HIDDEN_DIM; k++)
			{
				//Каждый элемент равен производной от функции активации для скрытого слоя, умноженному на частную производная ошибки E по массиву h1
				dE_dt1[k] = this->drelu(out[k]) * dE_dh1[k];
			}

			//Частная производная ошибки E по матрицу W1
			double* dE_dW1 = new double[INPUT_DIM * HIDDEN_DIM];

			//Транспонируем матрицу trainX, получаем trainX_transposed
			double* trainX_transposed = transposeMatrix(1, INPUT_DIM, trainX);
			
			//Перемножаем матрицы trainX_transposed на dE_dt1, записываем в dE_dW1
			multiplyMatrix(INPUT_DIM, HIDDEN_DIM, 1, trainX_transposed, dE_dt1, dE_dW1);

			//Умножаем полученные элементы градиента на шаг обучения ALPHA и корректируем веса на полученные значения
			//Для весов с входного на скрытый слой
			multiplyMatrixNumber(dE_dW1, INPUT_DIM * HIDDEN_DIM, ALPHA);
			diffMatrix(INPUT_DIM, HIDDEN_DIM, this->weights1, dE_dW1, this->weights1);
			
			//Для смещений с входного на скрытый слой
			multiplyMatrixNumber(dE_dt1, HIDDEN_DIM, ALPHA);
			diffMatrix(1, HIDDEN_DIM, this->bias1, dE_dt1, this->bias1);

			//Для весов со скрытого слоя на выходной слой
			multiplyMatrixNumber(dE_dW2, HIDDEN_DIM * OUT_DIM, ALPHA);
			diffMatrix(HIDDEN_DIM, OUT_DIM, this->weights2, dE_dW2, this->weights2);

			//Для смещений со скрытого слоя на выходной слой
			multiplyMatrixNumber(dE_dt2, OUT_DIM, ALPHA);
			diffMatrix(OUT_DIM, 1, this->bias2, dE_dt2, this->bias2);
			
			//Записываем значение ошибки в массив ошибок
			//В данной программе массив ошибок никак не используется, потенциально из него можно сделать график ошибок для отслеживания переобучения и других параметров
			this->losses[i + epoch * this->datasetSize] = E;

			//Освобождаем память
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
	* Метод подсчитывает процент ошибок с текущими весами в текущем датасете и выводит его на экран
	*/
	int count = 0;
	//Для каждого элемента датасета
	for (int i = 0; i < this->datasetSize; i++)
	{
		//Получить входные значения
		double* test_data = this->dataset[i];
		//Правильный ответ
		int correct = this->dataAnswers[i];
		//Получить предсказание от нейросети
		double* z = this->predict(test_data);
		//Преобразовать из вероятностного распределения в число
		int y_predict = argMax(z, 10);
		//Если предсказано точно
		if (y_predict == correct)
		{
			//Увеличить количество верно предсказанных чисел
			count += 1;
		}
	}
	//Подсчёт отношений правильно предсказанным к размеру датасета
	double accuracy = double(count) / double(datasetSize);
	cout << "Точность: " << accuracy << endl;
}

void Network::saveWeights()
{
	/*
	* Метод сохранения текущих весов и смещений в файлы w1, w2 и b1, b2 соответственно
	*/
	ofstream fout;
	fout.open("w1");
	if (!fout.is_open()) {
		cout << "Ошибка, не удалось открыть файл для сохранения весов" << endl;
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
	* Метод загрузки весов и смещений из файлов w1, w2 и b1, b2 соответственно
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
		cout << "Веса1 загружены" << endl;
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
		cout << "Смещения1 загружены" << endl;
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
		cout << "Веса2 загружены" << endl;
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
		cout << "Смещения2 загружены" << endl;
	}
	file.close();
}

void Network::showLayers()
{
	//Отладочный метод для отображения статистических данных о массивах
	statsArray(INPUT_DIM * HIDDEN_DIM, this->weights1);
	statsArray(HIDDEN_DIM, this->bias1);
	statsArray(HIDDEN_DIM * OUT_DIM, this->weights2);
	statsArray(OUT_DIM, this->bias2);
}