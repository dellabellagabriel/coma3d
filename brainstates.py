from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from pathlib import Path
import os

class StandardScaler3D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if X.ndim != 3:
            raise ValueError("X should have exactly 3 dimensions")
        return self

    def transform(self, X):
        standard_scaler = StandardScaler()
        X_reshape = X.reshape(X.shape[0], -1)
        X_standard = standard_scaler.fit_transform(X_reshape)

        return X_standard.reshape(X.shape[0], X.shape[1], X.shape[2])

class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=5, stride=1, tapering_function=None):
        self.window_size = window_size
        self.stride = stride
        self.tapering_function = tapering_function

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples, n_features, n_subs = X.shape
        n_windows = int((n_samples - self.window_size) / self.stride) + 1
        X_sliding = np.zeros((n_windows, self.window_size, n_features, n_subs))

        if self.tapering_function is not None:
            tapering_function = self.tapering_function(self.window_size, sym=False)
        else:
            tapering_function = np.repeat(1, self.window_size)

        for s in range(n_subs):
            for i in range(n_windows):
                start = i * self.stride
                end = start + self.window_size
                windowed_data = X[start:end, :, s]
                X_sliding[i, :, :, s] = windowed_data * tapering_function[:, np.newaxis]

        return X_sliding
    
class SubsamplerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_windows=1):
        self.n_windows = n_windows

    def fit(self, X, y=None):
        if X.shape[0] < self.n_windows:
            raise ValueError("n_windows can't be higher than the 1st dimension of X")
        return self

    def transform(self, X):
        X_subsample = X[::int(X.shape[0]/self.n_windows)]

        return X_subsample[:self.n_windows]
    
class CorrelationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_windows, n_time, n_rois, n_subs = X.shape
        X_corr = np.zeros((n_windows, n_rois, n_rois, n_subs))
        for i in range(n_windows):
            for j in range(n_subs):
                X_corr[i, :, :, j] = np.corrcoef(X[i,:,:,j].T)

        return X_corr
    

class AgglomerateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_trans = np.transpose(X, (3,0,1,2))
        n_subs, n_windows, n_rois, _ = X_trans.shape
        X_reshape = X_trans.reshape(-1, n_rois, n_rois)
        
        X_agg = np.zeros((n_subs*n_windows, n_rois*(n_rois-1)//2))
        for m in range(n_subs*n_windows):
            matrix = X_reshape[m,:].squeeze()
            X_agg[m, :] = matrix[np.triu_indices(n_rois, k=1)]

        return X_agg
    
    def inverse_transform(self, X):
        n_matrices, n_elements = X.shape
        n_rois = int((1 + np.sqrt(1 + 8 * n_elements)) / 2)
        X_res = np.zeros((n_matrices, n_rois, n_rois))
        for i in range(n_matrices):
            m = np.zeros((n_rois, n_rois))
            m[np.triu_indices(n_rois, k=1)] = X[i,:]
            m = m + m.T - np.diag(m.diagonal())
            X_res[i,:,:] = m

        return X_res

class ProbabilityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_subs, n_windows = X.shape
        prob = np.zeros((n_subs, self.n_clusters))
        for i in range(n_subs):
            for c in range(self.n_clusters):
                prob[i, c] = (X[i,:] == c).sum() / n_windows

        return prob

class BrainStates:
    """
    from_array: dictionary of conditions where each condition is time x rois x subject
    """
    def __init__(self, from_dict=None, from_path=None, output_path=None, export_vars=False, verbose=False):
        if from_dict is not None:
            self.conditions = list(from_dict.keys())
            self.conditions_idx = {}
            self.conditions_subs = {}
            self.verbose = verbose
            self.export_vars = export_vars
            idx = 0
            for key in from_dict.keys():
                condition_subs = from_dict[key].shape[2]
                self.conditions_idx[key] = list(range(idx, idx+condition_subs))
                self.conditions_subs[key] = condition_subs
                idx += condition_subs

            conditions_list = list(from_dict.values())
            self.data = np.concatenate(conditions_list, axis=2)
            self.has_run = False

        if not os.path.exists(output_path):
            Path(output_path).mkdir(parents=True, exist_ok=True)
            Path(f"{output_path}/figures").mkdir(parents=True, exist_ok=True)
            Path(f"{output_path}/statistics").mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Creating path {output_path} since it wasn't found...")

        self.output_path = output_path

        if verbose:
            print(self.summary())

    @staticmethod
    def export_mat(mat, name, path):
        sio.savemat(path, {name: mat})

    @staticmethod
    def import_mat(name, path):
        return sio.loadmat(path)[name]

    @staticmethod
    def sort_centroids(X):
        n_centroids, n_features = X.shape
        entropy = np.zeros(n_centroids)
        for i in range(n_centroids):
            hist, bin_edges = np.histogram(X[i, :], bins=int(np.sqrt(n_features)))
            norm_hist = hist / n_features
            entropy[i] = -np.sum(norm_hist * np.log2(norm_hist + 1e-12))
        
        idx = np.flip(np.argsort(entropy))
        return idx, entropy[idx], X[idx,:]
    
    def get_probs(self):
        result = {}
        for condition in self.conditions:
            result[condition] = self.probs[self.conditions_idx[condition],:]

        return result
    
    def get_centroids(self):
        return self.c_ord_matrix

    def calculate_prob(self, dfc):
        dfc_all_agg = self.clustering["agglomerate"].fit_transform(dfc)
        pred = cdist(dfc_all_agg, self.c_ord, metric="cityblock").argmin(axis=1)
        pred = pred.reshape(dfc.shape[3], dfc.shape[0])
        probs = ProbabilityTransformer(n_clusters=self.n_clusters).fit_transform(pred)

        return probs

    def summary(self):
        conditions = ""
        for condition in self.conditions:
            conditions += f"'{condition}' ({self.conditions_subs[condition]} subjects), "
        conditions = conditions[:-2]

        summary = f"There are {self.data.shape[2]} subjects total in {len(self.conditions)} conditions: {conditions}. Each with {self.data.shape[0]} samples and {self.data.shape[1]} regions. \n"
        if not self.has_run:
            summary += f"The analysis has not been run yet."
        else:
            summary += f"The analysis has been run with parameters:\n"
            summary += f"Window size = {self.window_size} samples\n"
            summary += f"Stride = {self.stride} samples\n"
            summary += f"Tapering function = {self.tapering_function}\n"
            summary += f"Subsampling = range divided into {self.subsampling} parts\n"
            summary += f"Number of clusters (k) = {self.n_clusters}\n"
            summary += f"Replicates = {self.n_init}\n"

        return summary

    def run(self, window_size, stride, tapering_function, subsampling, n_clusters, n_init, random_state=None):
        dfc_train_pipeline = Pipeline([
            ("scaler", StandardScaler3D()),
            ("sliding_window", SlidingWindowTransformer(window_size=window_size, stride=stride, tapering_function=tapering_function)),
            ("subsampler", SubsamplerTransformer(n_windows=subsampling)),
            ("correlation", CorrelationTransformer())
        ])

        dfc_all_pipeline = Pipeline([
            ("scaler", StandardScaler3D()),
            ("sliding_window", SlidingWindowTransformer(window_size=window_size, stride=stride, tapering_function=tapering_function)),
            ("correlation", CorrelationTransformer())
        ])

        clustering_pipeline = Pipeline([
            ("agglomerate", AgglomerateTransformer()),
            ("clustering", KMeans(n_clusters=n_clusters, n_init=n_init, init="k-means++", random_state=random_state))
        ])

        if self.verbose:
            print("Applying dFC transformations...")
        dfc_train = dfc_train_pipeline.fit_transform(self.data)
        dfc_all = dfc_all_pipeline.fit_transform(self.data)

        if self.verbose:
            print("Applying clustering pipeline...")
        clustering = clustering_pipeline.fit(dfc_train)
        c = clustering["clustering"].cluster_centers_
        idx, entropy, c_ord = self.sort_centroids(c)
        c_ord_matrix = AgglomerateTransformer().inverse_transform(c_ord)
        self.c_ord = c_ord
        self.c_ord_matrix = c_ord_matrix
        self.entropy = entropy
        self.clustering = clustering
        self.n_clusters = n_clusters

        if self.verbose:
            print("Calculating probabilities...")
        probs = self.calculate_prob(dfc_all)

        #outputs
        self.probs = probs
        self.dfc_all = dfc_all
        self.dfc_all_pipeline = dfc_all_pipeline
        
        #parameters
        self.window_size = window_size
        self.stride = stride
        self.tapering_function = tapering_function
        self.subsampling = subsampling
        self.n_init = n_init

        self.has_run = True
        
        if self.verbose:
            print("Exporting variables...")
        if self.export_vars:
            self.export_mat(dfc_all, "dfc", f"{self.output_path}/dfc.mat")
            self.export_mat(c_ord_matrix, "Cord", f"{self.output_path}/centroids.mat")
            self.export_mat(entropy, "H", f"{self.output_path}/entropy.mat")
            for condition in self.conditions:
                self.export_mat(self.probs[self.conditions_idx[condition],:], "probs", f"{self.output_path}/probs_{condition}.mat")


        if self.verbose:
            print(self.summary())