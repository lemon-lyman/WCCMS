import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from skimage.morphology import skeletonize_3d
from scipy import linalg, interpolate
import sys


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


class SkeletonNW:
    def __init__(self, bin_image):
        """

        :param bin_image: numpy array of the 3d binary image. Needs to be sufficiently processed. Smoothness of 3d image
        directly affects accuracy of skeleton
        """
        self.res_arr = np.array([96.39/512, 96.39/512, 157.6/197])
        self.raw_image = bin_image
        self.skeleton_image = skeletonize_3d(bin_image.astype(np.uint8)) ## Does the bulk of the 'skeletonizing'
        self.cell_xyz = self.get_cell_xyz()
        self.skel_points_pix = np.asarray(np.where(self.skeleton_image==self.skeleton_image.max())).T
        self.ax_equal_3d()
        self.skel_points = self.skel_points_pix*self.res_arr
        self.endpoints = []
        self.state = 0
        self.paths = []
        self.interp_paths = []
        self.delta = 100
        self.trash0 = None
        self.trash1 = None
        print("Initialized")

    def ax_equal_3d(self):
        """
        source - https://stackoverflow.com/a/13701747
        :return:
        """
        X = np.hstack((self.cell_xyz[:, 0], self.cell_xyz[:, 0].max()))
        Y = np.hstack((self.cell_xyz[:, 1], self.cell_xyz[:, 1].max()))
        Z = np.hstack((self.cell_xyz[:, 2], self.cell_xyz[:, 2].max()))
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

    def start_gui(self):
        self.initial_scatter = ax.scatter(self.skel_points[:, 0],
                                               self.skel_points[:, 1],
                                               self.skel_points[:, 2], s=30, c='k', picker=1)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_zlabel('Z (um)')
        self.title = ax.set_title("Select Endpoints")

    def check(self):
        """
        Checks the initial skeletonize_3d attempt. Call this or run the normal skeleton gui but not both
        :return:
        """
        self.initial_scatter = ax.scatter(self.skel_points[:, 0],
                                          self.skel_points[:, 1],
                                          self.skel_points[:, 2], s=10, c='r')
        self.cell_points = self.get_cell_xyz()
        ax.scatter(self.cell_points[::50, 0],
                   self.cell_points[::50, 1],
                   self.cell_points[::50, 2], s=3, c='b', alpha=.1)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_zlabel('Z (um)')

    def check_interp(self):
        """
        Checks the saved interpolated skeleton. Must be existing interpolated skeleton. Call this or run the normal
        skeleton gui but not both
        :return:
        """

        points = np.loadtxt("skeleton_temp/" + cell + "_points.txt", delimiter=',')

        self.initial_scatter = ax.scatter(points[:, 0],
                                          points[:, 1],
                                          points[:, 2], s=5, c='r')
        self.cell_points = self.get_cell_xyz()
        ax.scatter(self.cell_points[::5, 0],
                   self.cell_points[::5, 1],
                   self.cell_points[::5, 2], s=3, c='b', alpha=.03)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_zlabel('Z (um)')

    def get_cell_xyz(self):
        return np.asarray(np.where(self.raw_image == self.raw_image.max())).T * self.res_arr

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
        tck, u = interpolate.splprep(self.skel_points[self.paths[protrusion_idx]].T, k=5, w=w)
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

        ## Path to folder 'skeleton_temp' where skeleton data is saved
        ## Using 'cell' here as a global variable is probably bad practice
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
            self.cell_points = self.get_cell_xyz()
            ax.scatter(self.cell_points[::50, 0],
                       self.cell_points[::50, 1],
                       self.cell_points[::50, 2], s=3, c='b', alpha=.1)
            print("interp stored")



cell = sys.argv[1]
img = np.load("dumped_proccessed/" + cell + "_Post.npy") ## Path to NPY processed 3D binary image of cell

######################
skel = SkeletonNW(img)
######################

# skel.check_interp()

# skel.check()

######################
skel.start_gui()
fig.canvas.mpl_connect('pick_event', skel.onpick)
axnest = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnest, 'Next')
bnext.on_clicked(skel.next)
######################

plt.show()
