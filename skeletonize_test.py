import numpy as np
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import meshio


x_res = 141.7/512
y_res = x_res
z_res = .8

res_arr = np.array([x_res, y_res, z_res])

x_pix = 512
y_pix = x_pix
z_pix = 151

mesh = meshio.read('ProcessedMSHs/Gel4I.msh')

points_pix = mesh.points/res_arr
mins = np.floor(points_pix.min(0)).astype(int)
maxs = np.ceil(points_pix.max(0)).astype(int)

test_points = np.array(np.meshgrid(np.arange(mins[0], maxs[0]+1),
                                   np.arange(mins[1], maxs[1]+1),
                                   np.arange(mins[2], maxs[2]+1))).T.reshape(-1,3)

tri = Delaunay(points_pix)
tri.simplices = mesh.cells['triangle'].astype(np.int32)
output = np.asarray([tri.find_simplex(point) for point in test_points])
ind = np.argwhere(output>=0).flatten()
inside_points = test_points[ind]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], s=3, c='g')
plt.show()

