from scipy.signal import lfilter
from scipy import linalg
import numpy as np
from .complexity_pursuit_mask import return_mask
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly


class PointCloudCompressor:
    def __init__(self, time_serie, x_hat, y_hat, components_number):
        self.components = None
        self.time_serie = time_serie
        self.x_hat = x_hat
        self.y_hat = y_hat
        self.components_number = components_number
        self.fps = 30
        self.number_of_frames = time_serie.shape[0]
        self.time_axis = np.arange(self.number_of_frames) / self.fps

    def apply_pca(self):
        print('Apllying PCA in the phase series')
        print('Time serie shape', self.time_serie.shape)
        pca = PCA()
        eigen_vectors = pca.fit_transform(self.time_serie.T)
        eigen_values = pca.singular_values_
        components = pca.components_.T
        print("reduced matrix shape: ", pca.components_.shape)
        print("eigenvectors shape: ", eigen_vectors.shape, '\n')
        return eigen_vectors, eigen_values, components

    def apply_blind_source_separation(self, pca_components):
        components = pca_components[:, 0:self.components_number]
        print('Applying BSS')
        short_mask = return_mask(1.0, 10, 50)
        long_mask = return_mask(900000.0, 10, 50)
        print('calculating filters')
        short_filter = lfilter(short_mask, 1, components, axis=0)
        long_filter = lfilter(long_mask, 1, components, axis=0)
        print('Calculating covariance matrix')
        short_cov = np.cov(short_filter, bias=True, rowvar=False)
        long_cov = np.cov(long_filter, bias=True, rowvar=False)
        print('Calculating eigenvectors and eigenvalues')
        eigen_values, mixture_matrix = linalg.eig(long_cov, short_cov)
        print('mixing matrix shape: ', mixture_matrix.shape, '\n')
        mixture_matrix = np.real(mixture_matrix)
        unmixed = -np.matmul(components, mixture_matrix)
        unmixed = -np.flip(unmixed, axis=1)
        return mixture_matrix, unmixed

    def create_shapes_and_coordinates(self, eigen_vectors, mixture_matrix, sources):
        inverse_matrix = np.flip(np.linalg.inv(mixture_matrix), axis=0)
        mode_shapes = np.matmul(inverse_matrix, eigen_vectors[:, 0:self.components_number].T).T
        modal_coordinates = -sources
        return mode_shapes, modal_coordinates

    def plot_shapes_and_coordinates(self, modal_coordinates, mode_shapes):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        for column in range(self.components_number):
            ax = fig.add_subplot(2, 3, column + 1)
            ax.plot(self.time_axis, modal_coordinates[:, column], color="#069AF3")
        for column in range(self.components_number):
            mode_shape = mode_shapes[:, column]
            ax = fig.add_subplot(2, 3, column + 4, projection='3d')
            ax.set_zlim3d(bottom=-600, top=600)
            ax.scatter3D(self.x_hat, self.y_hat, mode_shape)
        plt.show()

        mode_shape = mode_shapes[:, 0]

        fig = go.Figure(data=[go.Scatter3d(x=self.x_hat, y=self.y_hat, z=mode_shape,
                                           mode='markers')])
        fig.update_layout(
            scene=dict(
                zaxis=dict(range=[-600, 600]),
            )
        )
        plotly.offline.plot(fig)

    def run(self):
        eigen_vectors, eigen_values, components = self.apply_pca()
        print("Eigen vector size: ", eigen_vectors.shape)
        mixture_matrix, sources = self.apply_blind_source_separation(components)
        mode_shapes, modal_coordinates = self.create_shapes_and_coordinates(eigen_vectors, mixture_matrix, sources)
        print(mode_shapes)
        print("Mode shape size arriving plotting: ", mode_shapes.shape, "\n")
        self.plot_shapes_and_coordinates(modal_coordinates, mode_shapes)
