import numpy as np
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from skimage.morphology import skeletonize_3d
from scipy import linalg, interpolate
import os, sys
from PIL import Image


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


class SkeletonNW:
    def __init__(self, bin_image):
        self.res_arr = np.array([141.7/512, 141.7/512, .8])

        self.skeleton_image = skeletonize_3d(bin_image.astype(np.uint8))
        self.skel_points_pix = np.asarray(np.where(self.skeleton_image==self.skeleton_image.max())).T
        self.skel_points = self.skel_points_pix*self.res_arr
        self.endpoints = []
        self.state = 0
        self.paths = []
        self.interp_paths = []
        self.trash0 = None
        self.trash1 = None

    def start_gui(self):
        self.initial_scatter = ax.scatter(self.skel_points[:, 0],
                                               self.skel_points[:, 1],
                                               self.skel_points[:, 2], s=30, c='k', picker=1)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_zlabel('Z (um)')
        self.title = ax.set_title("Select Endpoints")

    def onpick(self, event):
        ind = event.ind
        if len(ind) > 1:
            print("More than one data point was selected")
            ind = np.array([ind[0]])
        xdata, ydata, zdata = event.artist._offsets3d
        print(ind)
        if self.state == 0:
            self.endpoints.append(np.array([xdata[ind][0], ydata[ind][0], zdata[ind][0]]))
            ax.text(xdata[ind][0], ydata[ind][0], zdata[ind][0], str(len(self.endpoints) - 1))
            plt.draw()
            self.paths.append([ind[0]])
            print("Endpoint " + str(len(self.endpoints) - 1) + " selected")
        else:
            ax.plot([self.skel_points[self.paths[self.state-1][-1], 0], xdata[ind][0]],
                    [self.skel_points[self.paths[self.state-1][-1], 1], ydata[ind][0]],
                    [self.skel_points[self.paths[self.state-1][-1], 2], zdata[ind][0]], c='b')
            plt.draw()
            self.paths[self.state-1].append(ind[0])

    def interp(self, protrusion_idx):

        w = np.ones(len(self.paths[protrusion_idx]))*.1
        w[0] = w[-1] = 100
        tck, u = interpolate.splprep(self.skel_points[self.paths[protrusion_idx]].T, k=3, w=w)
        new = interpolate.splev(np.linspace(0,1,100), tck, der=0)
        ax.plot(new[0], new[1], new[2], c='r', linewidth=5)
        self.interp_paths.append(new)

    def store_interp(self):
        num_selected_points = 0
        point_chunks = []
        tangent_chunks = []
        self.num_per_protrusion = []
        for path in self.interp_paths:
            num_selected_points = num_selected_points + path[0].shape[0]
            point_chunk = np.concatenate((np.array([path[0]]).T,
                                          np.array([path[1]]).T,
                                          np.array([path[2]]).T,), axis=1)
            point_chunks.append(point_chunk)
            tangent_chunk = np.zeros((path[0].shape[0], 3))
            for t_idx, p_idx in zip(np.arange(1, path[0].shape[0]-1), np.arange(path[0].shape[0]-2, 0, -1)):
                tangent_chunk[t_idx, :] = np.vstack((point_chunk[p_idx, :] - point_chunk[p_idx+1, :],
                                                     point_chunk[p_idx-1, :] - point_chunk[p_idx, :])).mean(0)
            tangent_chunk[-1, :] = tangent_chunk[-2, :]
            tangent_chunk[0, :] = tangent_chunk[1, :]
            tangent_chunks.append(tangent_chunk)
            self.num_per_protrusion.append(path[0].shape[0])
        self.interp_points = np.zeros((num_selected_points, 3))
        self.interp_tangents = np.zeros((num_selected_points, 3))
        bookmark = 0
        for t_chunk, p_chunk in zip(tangent_chunks, point_chunks):
            self.interp_points[bookmark:bookmark+p_chunk.shape[0], :] = p_chunk
            self.interp_tangents[bookmark:bookmark+t_chunk.shape[0], :] = t_chunk
            bookmark = bookmark + t_chunk.shape[0]
            self.trash0 = p_chunk
            self.trash1 = t_chunk
        self.interp_tangents = (self.interp_tangents.T/np.linalg.norm(self.interp_tangents, axis=1)).T
        self.protrusion_log = []
        [[self.protrusion_log.append(ii) for _ in range(val)] for ii, val in enumerate(self.num_per_protrusion)]
        self.log_arr = np.asarray(self.protrusion_log)
        np.savetxt("skeleton_temp/" + cell + "_labels.txt", self.log_arr, delimiter=',')
        np.savetxt("skeleton_temp/" + cell + "_points.txt", self.interp_points, delimiter=',')
        np.savetxt("skeleton_temp/" + cell + "_tangents.txt", self.interp_tangents, delimiter=',')


    def next(self, event):
        self.state = self.state + 1
        if self.state <= len(self.endpoints):
            self.title.set_text("Protrusion " + str(self.state-1))
        elif self.state == len(self.endpoints) + 1:
            self.title.set_text("Interpolation")
            [self.interp(ii) for ii in range(len(self.endpoints))]
            plt.draw()
        elif self.state == len(self.endpoints) + 2:
            self.store_interp()
            print("interp stored")



cell = sys.argv[1]
img = np.load("dumped_proccessed/" + cell + "_Post.npy")
skel = SkeletonNW(img)

skel.start_gui()

fig.canvas.mpl_connect('pick_event', skel.onpick)
axnest = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnest, 'Next')
bnext.on_clicked(skel.next)
plt.show()