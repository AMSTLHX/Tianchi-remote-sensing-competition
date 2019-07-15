from osgeo import gdal
from PIL import Image
import os

if __name__ == '__main__':
    name = input("input the image number 1 or 2 you want clip:")
    imagepath = './data/image_{}.png'.format(name)
    n = os.path.basename(imagepath)[:-4]
    labelname = './data/' + n + '_label.png'
    dslb = gdal.Open(labelname)
    ds = gdal.Open(imagepath)
    wx = ds.RasterXSize
    wy = ds.RasterYSize
    stx = 0
    sty = 0
    step = 900
    outsize = 1500
    nullthresh = outsize * outsize * 0.7
    cx = 0
    cy = 0
    while cy + outsize < wy:
        cx = 0
        while cx + outsize < wx:
            img = ds.ReadAsArray(cx, cy, outsize, outsize)
            img2 = img[:3, :, :].transpose(1, 2, 0)
            if (img2[:, :, 0] == 0).sum() > nullthresh:
                cx += step
                print('kongbai...', cx, cy)
                continue

            img2 = Image.fromarray(img2, 'RGB')
            img2.save('./data/train/data1500/' + n + '_{}_{}.bmp'.format(cx, cy))
            # deal with label
            img = dslb.ReadAsArray(cx, cy, outsize, outsize)
            img = Image.fromarray(img).convert('L')
            img.save('./data/train/label1500/' + n + '_{}_{}.bmp'.format(cx, cy))

            cx += step
        cy += step