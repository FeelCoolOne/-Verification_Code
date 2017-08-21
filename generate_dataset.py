import os
os.environ["TF_CPP_MIN_LoG_LEVEL"]="2"

import numpy as np
from captcha.image import ImageCaptcha
from random import randint


TTF_DIR = "./ttf"
WORDS_LENS = 5
DATA_DIR = "./data"
width, height, n_len, nclass = 170, 80, 4, 10+2*26

A_Z=range(65, 91)
a_z=range(97, 123)
z_o=range(48, 57)
all_w = A_Z + a_z + z_o
i2c = {i: chr(ih) for i, ih in enumerate(all_w)}
c2i = {chr(ih):i for i, ih in enumerate(all_w)}


def text_generate(len_words):
    global all_w
    # random generate index of words
    indexs = list()
    for _ in range(len_words):
       indexs.append(randint(0, len(all_w)-1))
    
    l = [str(i2c[t]) for t in indexs]
    return ''.join(l)

def capcha_gen(fonts):
    image = ImageCaptcha(fonts=fonts, width=width, height=height)
    while True:
        label = text_generate(WORDS_LENS)
        img = image.generate_image(label)
        yield img, label

def image_generate(batch_size):
    fonts = filter(lambda x: x.endswith('.ttf'), os.listdir(TTF_DIR))
    assert len(fonts) != 0, "no fonts file"
    gen_func = capcha_gen(fonts)
    X = np.zeros((batch_size, height, width, 3), dtype="uint8")
    Y = np.zeros((batch_size, WORDS_LENS), dtype="uint8")
    for i in range(batch_size):
        img, label = next(gen_func)
        X[i] = img
        Y[i, :] = [c2i[j] for j in label]
    return X, Y

if __name__ == '__main__':
    # print(text_generate(5))
    X, Y =  image_generate(3)
    print X, Y

