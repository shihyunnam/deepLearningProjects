#https://en.wikipedia.org/wiki/Kernel_(image_processing)
#Convolution2D
import numpy as np
import scipy as sp
import itertools as it#iteration
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation
import warnings
# Suppress Matplotlib future warning
warnings.filterwarnings("ignore", category=FutureWarning)

# Your Matplotlib code here


def RGB_Convolution(img, kernel):
  img2 = np.copy(img)
  #loop over the rgb values
  for dim in range(3):#only iterate for rgb
    img2[:,:,dim] = sp.signal.convolve2d(img[:,:,dim],kernel, mode='same', boundary='symm')
  return img2
# def RGB_Convolution(img, kernel):
#     img2 = np.empty_like(img)

#     for dim in range(img.shape[-1]):
#         height, width = img.shape[:-1]

#         for i in range(height):
#             for j in range(width):
#                 img2[i, j, dim] = np.sum(img[i-1:i+2, j-1:j+2, dim] * kernel)

#     return img2
def RGB2RGBA(img, fill_value=1):
    # """Add an alpha channel to an RGB array"""
    if img.shape[-1] >= 4:#if channel has more or equal to 4
        return img
    img2 = np.full(shape=(*img.shape[:-1], 4),
                   fill_value=fill_value,
                   dtype=img.dtype)
    img2[:, :, :-1] = img
    return img2




plt.rcParams["figure.figsize"] = (14,7)#그래프 가로폭 세로폭 조정

#load an image
file_name = "/Users/shihyunnam/Desktop/pypractice/dog.jpg"
#we need to implement convolution with each pixel
kernels = {"edge_detection_kernel": np.array([[-1,-1,-1],
                                  [-1,8,-1 ],
                                  [-1,-1,-1]]),
           "sharpen_kernel": np.array([[0,-1,0],
                                  [-1,5,-1 ],
                                  [0,-1,0]]),
           "box_blur": (1/16) * np.array([[1,2,1],
                                  [2,4,2 ],
                                  [1,2,1]])}
kernel = kernels["edge_detection_kernel"]
#set img and imgFiltered
img_data = RGB2RGBA(plt.imread(file_name).astype(float) / 255)
# print(img_data)
# print(img_data.shape)#3 -> color channels(r,g,b)
# print(img_data.shape[-1])
img_filtered = RGB_Convolution(img_data, kernel)
img_display = np.copy(img_data)
# print(img_data[0])
# print(img_filtered[0])
fig, (axL,axR) = plt.subplots(ncols=2, tight_layout=True)
fig.suptitle(kernel)
imL = axL.imshow(img_data)
imR = axR.imshow(img_display)

# ////////////////////////////////////////////////
T = 10  # seconds
FPS = 30
FTOTAL = T*FPS  # total number of frames


indices = list(it.product(range(img_filtered.shape[0]), range(img_filtered.shape[1])))
Ninc = int(len(indices) / (FTOTAL))  # increment


def init_plot():
    axR.imshow(img_data)
    return (imR,)


def update(frame):
    for i in range(frame, frame + frame):
        if i >= len(indices):
            break
        idx_x, idx_y = indices[i]
        img_display[idx_x, idx_y, :] = img_filtered[idx_x, idx_y, :]
    imR.set_data(img_display)
    return (imR,)


if __name__ == "__main__":
    ani = FuncAnimation(fig, func=update, init_func=init_plot, interval=1000/FPS,
                         frames=range(0, len(indices), Ninc), repeat=False, blit=True)
    plt.show()
