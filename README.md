# Нейронная сеть для распознавания рукописных цифр
### Входные данные
- Массив из 784 (28x28) закодированных пикселей в 1 канале со значениями от 0 до 1 включительно

Для обучения использован датасет MNIST
Скачать необходимые файлы можно по адресу
https://huggingface.co/spaces/chrisjay/mnist-adversarial/tree/329b5253ff60239f7c1113985804eff8e3ac8216/files/MNIST/raw

### Модель нейронной сети
Для данной задачи была выбрана модель многослойного перцептрона, включающая:
- 784 нейрона на входном слою
- 128 нейронов на скрытом слою
- 10 нейронов на выходном слою
![Нейронные сети, перцептрон — Викиконспекты](https://neerc.ifmo.ru/wiki/images/thumb/6/63/Multi-layer-neural-net-scheme.png/500px-Multi-layer-neural-net-scheme.png)
Алгоритм обучения:
- Градиентный спуск
Алгоритм вычисления градиента:
- Обратное распространение

Код нейросети написан на чистом **c++**

### Файловая структура:
	- Source.cpp - главный файл программы
	- Network.h - Файл заголовков нейросети
	- Network.cpp - Файл реализации нейросети
	- Tools.h - Файл заголовков вспомогательных методов (перемножение, транспонирование матриц и тп.)
	- Tools.cpp - Файл реализации вспомогательных методов
	- train-images.idx3-ubyte - Датасет изображений для тренировки
	- train-labels.idx1-ubyte - Датасет правильных ответов для тренировки
	- t10k-images.idx3-ubyte - Датасет изображений для тестирования
	- t10k-labels.idx1-ubyte - Датасет правильных ответов для тестирования
	- convert.py - Скрипт для конвертации изображений на python
	- requirements.txt - Зависимости для работы скрипта на python

### Предсказание цифры с картинки

Для предсказания цифры, изображённой на картинке требуется:
- Заиметь картинку 28*28 пикселей в 1 канале (чёрно-белую)
- Закодировать пиксели в массив чисел от 0 до 1, записать их в файл image, каждый пиксель с новой строки
- Вызвать в методе main файла Source.cpp процедуру train, процедуру predictTest
### Конвертация изображения при помощи скрипта на python

Для того, чтобы использовать скрипт convert.py, находящийся в папке проекта требуется:
- иметь установленный python3 с библиотеками pillow, numpy
- Вставить картинку в папку Project8, назвав её test.png
- Выполнить команду `python convert.py` из папки Project8
После конвертации можно вызывать Source.cpp, в main-е которого вызывается процедура predictTest