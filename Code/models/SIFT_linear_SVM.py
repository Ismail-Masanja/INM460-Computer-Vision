import cv2
import torch
import numpy as np
from torch import nn

from utils.helpers import calculate_sift_features, image_preprocess


__all__ = ['SIFTLinearSVM']


class SIFTLinearSVM(nn.Module):
    """Linear Support Vector Machine  Combined with Kmeans and SIFT"""

    def __init__(self, kmeans_model, num_classes, device):
        super(SIFTLinearSVM, self).__init__()
        self.device = device
        self.feature_extractor = SIFTKMeansFeatureExtractor(kmeans_model, device)
        self.k = kmeans_model.n_clusters # Number of Clusters
        
        # Initialize weights and bias with correct shape
        self.W = nn.Parameter(torch.randn(num_classes, self.k, device=self.device), requires_grad=True)
        self.B = nn.Parameter(torch.randn(num_classes, device=self.device), requires_grad=True)

    def forward(self, X):
        # Extract SIFT Features and predict Clusters
        X = self.feature_extractor(X)
           
        # Linear multiplication with weights and bias
        output = X.matmul(self.W.t()) + self.B
        return output

        
class SIFTKMeansFeatureExtractor(nn.Module):
    def __init__(self, kmeans_model, device):
        super(SIFTKMeansFeatureExtractor, self).__init__()
        self.kmeans = kmeans_model
        self.device = device

    def forward(self, batch):
        batch_size = batch.size(0)
        hist_features = torch.zeros((batch_size, self.kmeans.n_clusters), device=self.device)
        for i in range(batch_size):
            # Convert PyTorch tensor to numpy array and prepare SIFT
            img_np = batch[i].permute(1, 2, 0).cpu().numpy()  # Input tensor [Channels, Height, Width]
            img_np = image_preprocess(img_np)

            # Extract SIFT features
            _, descriptors = calculate_sift_features(img_np)
            
            # Histogram of cluster assignments
            if descriptors is not None:
                hist = np.zeros(self.kmeans.n_clusters)
                idx = self.kmeans.predict(descriptors)
                for j in idx:
                    hist[j] += 1
                hist = hist / (np.linalg.norm(hist) + 1e-6)  # L2 normalization
                features = torch.from_numpy(hist).to(self.device).float()
            else:
                features = torch.zeros(self.kmeans.n_clusters, device=self.device)

            hist_features[i] = features
            
        return hist_features