import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceCNN(nn.Module):
    """
    Convolutional Neural Network with normalized Cosine Metric Head.
    Maps 96x96 grayscale face crops into a unit 128-D embedding space.
    Cosine similarity against class centroids guarantees accurate unknown face detection.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 96 -> 48

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Class weight vectors for Cosine Similarity
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def extract_features(self, x):
        """Extract L2-normalized 128-D embedding vector for input face."""
        feat = self.features(x)
        embed = self.embedding(feat)
        return F.normalize(embed, p=2, dim=1)

    def forward(self, x, scale=16.0):
        """Returns scaled logits for CrossEntropyLoss and raw cosine similarities."""
        embed_norm = self.extract_features(x)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embed_norm, weight_norm)
        logits = cosine * scale
        return logits, cosine