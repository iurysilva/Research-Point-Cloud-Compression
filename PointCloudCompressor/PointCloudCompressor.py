from scipy.signal import lfilter
from scipy import linalg
import numpy as np
from .complexity_pursuit_mask import return_mask
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PointCloudCompressor:
    def __init__(self, time_serie, components_number):
        self.components = None
        self.time_serie = time_serie
        self.components_number = components_number
        self.fps = 30
        self.number_of_frames = time_serie.shape[0]
        self.time_axis = np.arange(self.number_of_frames) / self.fps

    def apply_pca(self):
        print('Apllying PCA in the phase series')
        pca = PCA()
        eigen_vectors = pca.fit_transform(self.time_serie.T)
        print("reduced matrix shape: ", pca.components_.shape)
        print("eigenvectors shape: ", eigen_vectors.shape, '\n')
        return eigen_vectors.T, pca.singular_values_, pca.components_.T

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

    def run(self):
        eigen_vectors, eigen_values, components = self.apply_pca()
        mixture_matrix, sources = self.apply_blind_source_separation(components)
        mode_shapes, modal_coordinates = self.create_shapes_and_coordinates(eigen_vectors, mixture_matrix, sources)
        fig2, axs2 = plt.subplots(2, self.components_number)
        for column in range(self.components_number):
            axs2[0][column].plot(self.time_axis, modal_coordinates[:, column], color="#069AF3")
        plt.show()
