# 默认按照3×3的大小切割图像保存到指定文件夹
from PIL import Image

def cut_picture(path_and_picture, dst_path, xx = 3, yy = 3):
    im = Image.open(path_and_picture)
    # 图片的宽度和高度
    img_size = im.size
    print("图片宽度和高度分别是{}".format(img_size))

    x = img_size[0] // xx
    y = img_size[1] // yy
    for j in range(yy):
        for i in range(xx):
            left = i * x
            up = y * j
            right = left + x
            low = up + y
            region = im.crop((left, up, right, low))
            print((left, up, right, low))
            temp = str(i) + str(j)
            region.save(dst_path + "/" + temp + ".jpg")
