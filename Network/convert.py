from PIL import Image
import numpy as np

im_frame = Image.open('test.png')
np_frame = np.array(im_frame.getdata())
if (np_frame.shape != (784, 1)):
    print("Конвертация возможна только для одноканальных изображений 28x28 пикселей")
else:
    t1 = np.full(784, 1)
    if (abs((max(t1) - 1)) < 0.001):
        np.savetxt("image", np_frame)
    else:
        np.savetxt("image", np_frame / 255)
    print("Изображение конвертировано в файл image")
