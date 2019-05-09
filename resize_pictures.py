import os
import cv2

# 将.jpg格式文件调整大小后存入到dst_path中
def resize_pic(src_path, dst_path, sizes_ = (256,256)):

    for item in os.listdir(src_path):
        if item.endswith('.jpg'):
            pic_path=os.path.join(src_path, item)
            img=cv2.imread(pic_path)
            dst=cv2.resize(img, sizes_)
            ddst_path=os.path.join(dst_path, item)
            cv2.imwrite(ddst_path, dst)