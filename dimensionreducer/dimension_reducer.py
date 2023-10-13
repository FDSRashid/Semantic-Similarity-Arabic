from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import DictionaryLearning
import numpy as np
from scipy.stats import skew, kurtosis
from umap import UMAP


class DimensionReducer:
    """
    A class for reducing the dimensionality of embedded sentences or high-dimensional data using various methods.
    
    Parameters:
    - method (str): The dimensionality reduction method to use. Supported methods include 'PCA', 't-SNE',
      'UMAP', 'KernelPCA', 'RandomProjection', 'PCA_SVD', 'DictionaryLearning', and 'KMeans'.
    - n_components (int): The number of components or dimensions to reduce the data to. Note- for KMEans, 
      you should provide the number of clusters to group the sentences to.
    - model_args (dict): Additional arguments to pass to the dimensionality reduction model (optional).

    Attributes:
    - method (str): The chosen dimensionality reduction method.
    - n_components (int): The number of components to reduce data to.
    - model_args (dict): Additional arguments for the dimensionality reduction model.
    - reduced_data (numpy.ndarray): The reduced representation of the data.

    Methods:
    - read_list(path: str) -> numpy.ndarray:
      Reads a CSV file containing embedded sentences and returns a numpy array.

    - analyze_sample(embedded_data: numpy.ndarray) -> dict:
      Analyzes the input data and provides various characteristics, such as data statistics, standardization,
      scaling, similarity matrix, skewness, and kurtosis.

    - reduce_dimension(embedded_data: numpy.ndarray):
      Reduces the dimensionality of the input data based on the chosen method and stores the result in
      the 'reduced_data' attribute.

    - get_reduced_data() -> numpy.ndarray:
      Retrieves the reduced data. Raises an error if data has not been reduced using 'reduce_dimension()'.
    """

    def __init__(self, method='PCA', n_components=2, model_args=None):
        available_methods = ['PCA', 't-SNE', 'UMAP', 'KernelPCA', 'RandomProjection', 'PCA_SVD',
                             'DictionaryLearning', 'KMeans']
        if method not in available_methods:
            raise ValueError(f"Unsupported method '{method}'. Supported methods are: {', '.join(available_methods)}")
        self.method = method
        self.n_components = n_components
        self.model_args = model_args if model_args is not None else {}
        self.reducer = None
        self.reduced_data = None  # Placeholder for reduced data
    def read_list(self, path):
        """
        Reads a CSV file containing embedded sentences and returns a numpy array.

        Parameters:
        - path (str): The path to the CSV file containing embedded sentences.

        Returns:
        - numpy.ndarray: A numpy array representing the embedded sentences.
        """
        if not isinstance(path, str):
            raise ValueError('Input must be a string.')
        data = np.genfromtxt(path, delimiter = ',', dtype = str)
        data = [eval(row) for row in data]
        return np.array(data)
    
    def analyze_sample(self, embedded_data):
        """
        Analyzes the input data and provides various characteristics.
        Remember, for Dimensionality Reduction, you want a sample size about 5-10 ten times the number
        of dimensions each sample has. In our case, for embedded sentences, which generally is a 768 feature vector,
        we want a 3500 - 8000 sample size. The scaling and Standardized, feature magnitude, and similarity
        components are done on the unedited data. if you want correlation of the standardized data, save it by calling the function first
        and save results to a variable, then reinput the function using one of the items of the original call.

        Parameters:
        - embedded_data (numpy.ndarray): The input data containing embedded sentences.

        Returns:
        - dict: A dictionary containing data characteristics, statistics, and more.
        """
        report = {}
        if  not isinstance(embedded_data, np.ndarray):
            raise ValueError('Not a numpy array. Please input numpy array or use read_list() if needed')
        numdims = embedded_data.shape[1]
        n = embedded_data.shape[0]
        report['Data Characteristics'] = {
        'Number of Samples': n,
        'Number of Dimensions (Features)': numdims}
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(embedded_data)
        report['Standardized Data'] = standardized_data
        feature_magnitudes = np.linalg.norm(embedded_data, axis=1)
        report['Feature Magnitude'] = feature_magnitudes
        scaler2 = MinMaxScaler()
        scaled_data = scaler2.fit_transform(embedded_data)
        report['Scaled Data'] = scaled_data
        correlation_matrix = np.corrcoef(embedded_data, rowvar=False)
        report['Similarity'] = correlation_matrix
        skewness = skew(embedded_data, axis=1)  # Calculate skewness for each feature
        kurt = kurtosis(embedded_data, axis=1)
        report['Skewness'] =skewness
        report['Kurtosis'] = kurt
        return report





    def reduce_dimension(self, embedded_data):
        """
        Reduces the dimensionality of the input data based on the chosen method.

        Parameters:
        - embedded_data (numpy.ndarray): The input data containing embedded sentences.
        """
        if self.method == 'PCA':
            self.reducer = PCA(n_components=self.n_components, **self.model_args)
        elif self.method == 't-SNE':
            self.reducer = TSNE(n_components=self.n_components, **self.model_args)
        elif self.method == 'UMAP':
            self.reducer = UMAP(n_components = self.n_components, **self.model_args)
        elif self.method == 'KernelPCA':
            self.reducer = KernelPCA(n_components=self.n_components, **self.model_args)
        elif self.method == 'RandomProjection':
            self.reducer = GaussianRandomProjection(n_components=self.n_components, **self.model_args)
        elif self.method == 'PCA_SVD':
            self.reducer = PCA(n_components=self.n_components, svd_solver = 'randomized', **self.model_args)
        elif self.method == 'DictionaryLearning':
            self.reducer = DictionaryLearning(n_components=self.n_components, **self.model_args)
        elif self.method == 'KMeans':
            # Perform K-Means clustering to get cluster assignments
            kmeans = KMeans(n_clusters=self.n_components)
            cluster_labels = kmeans.fit_predict(embedded_data)
            self.reduced_data = cluster_labels  # Use cluster labels as the reduced representation
            return
        self.reduced_data = self.reducer.fit_transform(embedded_data)
        
    

    def get_reduced_data(self):
        """
        Retrieves the reduced data.

        Returns:
        - numpy.ndarray: The reduced representation of the data.
        """
        if self.reduced_data is None or self.reduced_data.size == 0:
            raise ValueError('Data has not been reduced. Please reduce data first using reduce_dimension()')
            
        return self.reduced_data
