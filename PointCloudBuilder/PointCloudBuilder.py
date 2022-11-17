from pyntcloud import PyntCloud
import copy
import scipy.stats as stats
import scipy
from .polynomial_regression import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PointCloudBuilder:
    def __init__(self, pstat, pstat2, filename_prefix, avg_pc, n_acc, n_files, init_pc):
        self.jj = 0
        self.ff = 1
        self.ff2 = 4
        self.pstat = pstat
        self.pstat2 = pstat2
        self.filename_prefix = filename_prefix
        self.avg_pc = avg_pc
        self.n_acc = n_acc
        self.ss = np.random.randint(1, avg_pc, (n_acc, 1))
        self.n_files = n_files
        self.init_pc = init_pc
        self.x_hat = None
        self.y_hat = None
        self.displacements = None
        self.inpt_hat = None
        self.z_hat_old = None
        self.iteration = init_pc

        # Matplotlib global variables
        self.scat3d = 0
        self.anim = 0
        self.ax1 = plt.axes(projection='3d')

    def build_one_frame(self, frame=False):
        point_cloud_name = self.filename_prefix + str(self.iteration) + '.ply'
        print("Reading point cloud %d: %s..." %(self.jj, point_cloud_name))
        point_cloud = PyntCloud.from_file(point_cloud_name)
        print(point_cloud.points.shape)
        print("Copying point cloud...")
        point_cloud2 = copy.deepcopy(point_cloud)

        # Cropping
        print("Cropping...")
        location = np.array(point_cloud.points)[:, 0:3]
        indices = (location[:, 0] <= np.max(self.pstat[:, 0])) & (location[:, 0] >= np.min(self.pstat[:, 0])) & \
                  (location[:, 1] <= np.max(self.pstat[:, 1])) & (location[:, 1] >= np.min(self.pstat[:, 1])) & \
                  (location[:, 2] <= np.max(self.pstat[:, 2])) & (location[:, 2] >= np.min(self.pstat[:, 2]))
        indices = np.where(indices == 1)[0]
        point_cloud.apply_filter(indices)

        # Cropping again
        location = np.array(point_cloud2.points)[:, 0:3]
        indices = (location[:, 0] <= np.max(self.pstat2[:, 0])) & (location[:, 0] >= np.min(self.pstat2[:, 0])) & \
                  (location[:, 1] <= np.max(self.pstat2[:, 1])) & (location[:, 1] >= np.min(self.pstat2[:, 1])) & \
                  (location[:, 2] <= np.max(self.pstat2[:, 2])) & (location[:, 2] >= np.min(self.pstat2[:, 2]))
        indices = np.where(indices == 1)[0]
        point_cloud2.apply_filter(indices)

        print("Centering Xs and Ys...")
        # unpacking point cloud x, y and y values
        location = np.array(point_cloud2.points)[:, 0:3]
        x = location[:, 0]
        y = location[:, 1]
        z = location[:, 2]
        count_sensors = np.array(point_cloud.points.shape[0])
        inpt = np.round(stats.zscore(np.column_stack([x, y])), decimals=4)

        # creating the numerical model for getting the sensors and displacements
        print("Creating polinomial model...")
        if self.jj == 0:
            self.inpt_hat = inpt[self.ss, :]
            fitted_parameters, p_cov = scipy.optimize.curve_fit(func, inpt, z)
            self.x_hat = x[np.ravel(self.ss)]
            self.y_hat = y[np.ravel(self.ss)]
            color = np.array(point_cloud2.points)[:, 3:6]
        else:
            fitted_parameters, p_cov = scipy.optimize.curve_fit(func, inpt, z)

        print("Applying model...")    
        z_hat = func_array(self.inpt_hat, fitted_parameters[0], fitted_parameters[1], fitted_parameters[2],
                           fitted_parameters[3], fitted_parameters[4])
        aux = np.array([self.x_hat, self.y_hat, z_hat])

        print("Creating and saving displacements...")
        if self.jj > 0:
            self.displacements[self.jj] = aux - self.z_hat_old
        else:
            self.displacements = np.zeros((self.n_files - self.init_pc, 3, aux.shape[1]))
            self.displacements[self.jj] = np.zeros(aux.shape)
        self.z_hat_old = aux

        print("Incrementing variables...")
        self.ff += 1
        self.jj += 1
        self.iteration += 1

        print("Plotting scatter, saving figure...")
        self.scat3d.remove()
        self.scat3d = self.ax1.scatter3D(self.x_hat, self.y_hat, z_hat, c="gray")
        plt.savefig("plots/"+("img%d.jpg" % (self.iteration-40)))
        if self.iteration >= self.n_files:
            np.save('arrays/displacements', self.displacements)
            return self.displacements
        print("Point cloud %s finalized! \n" % self.jj)

    def run(self, animation_velocity=12000):
        self.scat3d = self.ax1.scatter3D(0, 0, 0)
        self.ax1.view_init(50, 270)
        self.ax1.set_xlabel('X (mm)')
        self.ax1.set_ylabel('Y (mm)')
        self.ax1.set_zlabel('Z (mm)')
        self.ax1.set_zlim3d(bottom=900, top=1250)
        print("Initializing with 3D animation")
        while True:
            self.build_one_frame()

        # self.anim = FuncAnimation(plt.gcf(), self.build_one_frame, interval=animation_velocity, repeat=False)
        # plt.show()
