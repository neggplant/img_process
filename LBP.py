# 原始LBP，周围8个像素点，判断与中心点的大小关系，按照pow(2, i)进行求和
def LBP(src):
    '''
    param src:灰度图像
    return：返回特征图像
    '''
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    
    lbp_value = np.zeros((1,8), dtype=np.uint8)
    neighbours = np.zeros((1,8), dtype=np.uint8)
    for x in range(1, width-1):
        for y in range(1, height-1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128

            dst[y, x] = lbp

    return dst
    
def circular_LBP(src, radius, n_points):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    src.astype(dtype=np.float32)
    dst.astype(dtype=np.float32)

    neighbours = np.zeros((1, n_points), dtype=np.uint8)
    lbp_value = np.zeros((1, n_points), dtype=np.uint8)
    for x in range(radius, width - radius - 1):
        for y in range(radius, height - radius - 1):
            lbp = 0.
            # 先计算共n_points个点对应的像素值，使用双线性插值法
            for n in range(n_points):
                theta = float(2 * np.pi * n) / n_points
                x_n = x + radius * np.cos(theta)
                y_n = y - radius * np.sin(theta)

                # 向下取整
                x1 = int(math.floor(x_n))
                y1 = int(math.floor(y_n))
                # 向上取整
                x2 = int(math.ceil(x_n))
                y2 = int(math.ceil(y_n))

                # 将坐标映射到0-1之间
                tx = np.abs(x - x1)
                ty = np.abs(y - y1)

                # 根据0-1之间的x，y的权重计算公式计算权重
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbour = src[y1, x1] * w1 + src[y2, x1] * w2 + src[y1, x2] * w3 + src[y2, x2] * w4

                neighbours[0, n] = neighbour

            center = src[y, x]

            for n in range(n_points):
                if neighbours[0, n] > center:
                    lbp_value[0, n] = 1
                else:
                    lbp_value[0, n] = 0

            for n in range(n_points):
                lbp += lbp_value[0, n] * 2**n

            # 转换到0-255的灰度空间，比如n_points=16位时结果会超出这个范围，对该结果归一化
            dst[y, x] = int(lbp / (2**n_points-1) * 255)

    return dst
