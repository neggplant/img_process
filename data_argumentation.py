from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def data_argumentation(path_picture, dir_save_, numbers_=20, save_prefix = "img", format_img = "png"):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.8,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(path_picture)  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(width, high, channals)
    x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, width, high, channals)
    # 生产的所有图片保存在 save_to_dir 目录下
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=dir_save_,
                              save_prefix=save_prefix,
                              save_format=format_img,
                              seed=4):
        i += 1
        if i > numbers_:
            break  # 否则生成器会退出循
