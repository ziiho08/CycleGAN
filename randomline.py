import numpy as np
import matplotlib.pyplot as plt
import os
import natsort
import cv2

# Bezier Curve 사용
def recta(x1, y1, x2, y2):
    if x1 == x2:  # 분모가 0이되는거 방지
        x2 -= 1
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    return (a, b)


def curva_1(xa, ya, xb, yb, xc, yc, parameter):
    (x1, y1, x2, y2) = (xa, ya, xb, yb)
    (a1, b1) = recta(xa, ya, xb, yb)
    (a2, b2) = recta(xb, yb, xc, yc)
    puntos = []

    for i in range(0, parameter):
        if x1 == x2:
            continue
        else:
            (a, b) = recta(x1, y1, x2, y2)
        x = i * (x2 - x1) / parameter + x1
        y = a * x + b
        #         puntos.append((x,y))
        puntos.append(x)
        x1 += (xb - xa) / parameter
        y1 = a1 * x1 + b1
        x2 += (xc - xb) / parameter
        y2 = a2 * x2 + b2
    return puntos


def curva_2(xa, ya, xb, yb, xc, yc, parameter):
    (x1, y1, x2, y2) = (xa, ya, xb, yb)
    (a1, b1) = recta(xa, ya, xb, yb)
    (a2, b2) = recta(xb, yb, xc, yc)
    puntos = []

    for i in range(0, parameter):
        if x1 == x2:
            continue
        else:
            (a, b) = recta(x1, y1, x2, y2)
        x = i * (x2 - x1) / parameter + x1
        y = a * x + b
        puntos.append((x, y))
        #         puntos.append(x)
        x1 += (xb - xa) / parameter
        y1 = a1 * x1 + b1
        x2 += (xc - xb) / parameter
        y2 = a2 * x2 + b2
    return puntos


def draw_data(parameter, num_data):
    length = len(os.listdir('./make_line/'))
    for i in range(num_data):
        # random point 생성
        x = np.random.randint(128)
        y = np.random.randint(128)
        c1 = np.random.randint(128 - x)
        c2 = np.random.randint(128 - y)
        point1 = np.random.randint(0, 128), 0  # x-axis
        point2 = 64, np.random.randint(0, 128)  # y-axis
        point3 = c1, c2

        lista1 = curva_1(point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], parameter)
        lista2 = curva_2(point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], parameter)

        fig, ax = plt.subplots()
        x = 166 / fig.dpi
        y = 170 / fig.dpi
        fig.set_figwidth(x)
        fig.set_figheight(y)
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        # print(lista1)
        ax.plot(lista1, color='w')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.savefig(f'./make_line/{length + i}_draw.png', facecolor='k', transparent=True, bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        # print(lista1)
        fig, ax = plt.subplots()
        x = 166 / fig.dpi
        y = 170 / fig.dpi
        fig.set_figwidth(x)
        fig.set_figheight(y)
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        # print(lista1)
        ax.plot(lista2, color='w')

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.savefig(f'./make_line/{length + num_data + i}_draw.png', facecolor='k', transparent=True,
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    for j in range(num_data):
        # random point 생성
        a = np.random.randint(128)
        b = np.random.randint(128)
        c1 = np.random.randint(128 - a)
        c2 = np.random.randint(128 - b)
        point1 = np.random.randint(0, 128), 0  # x-axis
        point2 = 64, np.random.randint(0, 128)  # y-axis
        point3 = c1, c2

        lista1 = curva_1(point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], parameter)
        lista2 = curva_2(point1[0], point1[1], point2[0], point2[1], point3[0], point3[1], parameter)
        fig, ax = plt.subplots()
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        x = 166 / fig.dpi
        y = 170 / fig.dpi
        fig.set_figwidth(x)
        fig.set_figheight(y)
        ax.plot(lista1, color='w')

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.savefig(f'./make_line/{(length + num_data * 2) + j}_draw.png', facecolor='k', transparent=True,
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        fig, ax = plt.subplots()
        plt.xlim(0, 128)
        plt.ylim(0, 128)
        # print(lista1)
        x = 166 / fig.dpi
        y = 170 / fig.dpi
        fig.set_figwidth(x)
        fig.set_figheight(y)
        ax.plot(lista2, color='w')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.savefig(f'./make_line/{length + num_data * 3 + j}_draw.png', facecolor='k', transparent=True,
                    bbox_inches='tight', pad_inches=0)
        plt.close()

for parameter in [150, 200, 250, 300, 400, 450, 500, 550, 600]:
    print(parameter)

    draw_data(parameter, num_data=100)

    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets, models, transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import natsort
    import cv2


    class processing(Dataset):
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.image_fns = natsort.natsorted(os.listdir(data_dir))  # image 정렬

        def __len__(self):
            return len(self.image_fns)

        def __getitem__(self, index):
            image_fn = self.image_fns[index]  # image 하나의 파일 명 불러오기
            image_fp = os.path.join(self.data_dir, image_fn)  # 하나의 image 파일 path
            image = cv2.imread(image_fp, cv2.IMREAD_COLOR)

            return image

    path = './make_line/'
    image_set = processing(path)
    image_loader = DataLoader(image_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    for idx, img in enumerate(image_loader):
        img = img[0, :, :, :]
        img = np.array(img)
        cv2.imwrite(f'./dataset/make_line/{idx + 1}_.png', img)
