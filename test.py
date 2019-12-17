import cv2
import imageio
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from PIL import Image
from sklearn import datasets
from sklearn.cluster import KMeans

# arr = np.array([(0, 0, 0, 4, 5, 6, 0, 0), (0, 0, 0, 4, 5, 6, 0, 0)])
# nRow, nCol = arr.shape[:2]
#
# for i in range(1, nCol):
#     if arr[1][i] != 0 and arr[1][i-1] == 0:
#         first = arr[1][i]
#         break
# last = 0
# for i in range(1, nCol-1):
#     if arr[1][nCol - i] != 0 and arr[1][nCol - i + 1] == 0:
#         last = arr[1][nCol - i]
#         break
#
# print(last)

padding = []
for i in range(10):
    padding.append([0])

print(len(padding[0]))