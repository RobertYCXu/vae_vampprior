import cv2
from glob import glob
import os
import numpy as np

def genimg(dirname, padding=4):
    imgs = glob(dirname + '/*')

    imgs = sorted(imgs)

    imgs = [cv2.imread(img) for img in imgs]
    try:
        imh, imw, imc = imgs[0].shape
    except Exception as e:
        print('eh')

    vpadding = np.full((padding, imw, 3), 255, dtype=np.uint8)

    result = []
    for i in range(4):
        stacked_result = np.ndarray((7,), dtype=np.ndarray)
        for k in range(4):
            stacked_result[k * 2] = imgs[i + 4 * k]
        stacked_result[1] = vpadding
        stacked_result[3] = vpadding
        stacked_result[5] = vpadding

        result.append(np.vstack(stacked_result))

    alh = result[0].shape[0]
    hpadding = np.full((alh, padding, 3), 255, dtype = np.uint8)
    stacked = [result[0], hpadding, result[1], hpadding, result[2], hpadding, result[3]]

    result = np.hstack(stacked)

    return result



if __name__ == '__main__':
    dirs = glob('./generate/*')

    out = './gen_out'
    if not os.path.exists(out):
        os.makedirs(out)

    for dir in dirs:
        dirname = os.path.basename(dir)
        result = genimg(dir)

        cv2.imwrite(os.path.join(out, dirname + ".jpg"), result)



