import os

path = ""

# rename pictures in path,
def rename_pic(path, name_dst = 'img_', init_index = 1):

    # 将图片首先转换成其他格式文件，预防转换时删除重复的名字文件
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(path, item +'o')
        try:
            os.rename(src, dst)
        except:
            continue

    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(path, name_dst + str(init_index) + '.jpg')
        try:
            os.rename(src, dst)
            print("convert %s to %s" % (src, dst))
            init_index = init_index + 1
        except:
            continue
