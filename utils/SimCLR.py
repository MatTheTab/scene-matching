import torch
import torchvision.models as models
import numpy as np
import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import STL10
from torch.multiprocessing import cpu_count
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import SGD, Adam
import random
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.decomposition import PCA
from tqdm import tqdm


BATCH_SIZE = 32
WEIGHT_DECAY = 1e-6
MAX_EPOCHS = 100
OPTIMIZER = "adam"
LR = 3e-4
GRADIENT_ACCUMULATION_STEPS = 5
MLP_DIM = 512
EMBEDDING_SIZE_LARGE = 64
EMBEDDING_SIZE_SMALL = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

class SynchronizedAugment:
    """
    Apply the same random transformation across a list of PIL images.
    Useful when you have multi-view data and want consistent augmentation per sample.
    """
    def __init__(self, img_size, s=1):
        self.img_size = img_size
        self.blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        self.s = s

    def __call__(self, imgs):
        """
        imgs: list of PIL.Image (length 5)
        Returns:
            torch.Tensor of shape [5, 224, 224]
        """
        # Random parameters
        scale = random.uniform(0.6, 1.0)
        flip = random.random() < 0.5
        apply_blur = random.random() < 0.3

        # Synchronized crop
        i, j, h, w = T.RandomResizedCrop.get_params(imgs[0], scale=(scale, 1.0), ratio=(1.0, 1.0))

        processed = []
        for img in imgs:
            img = TF.resized_crop(img, i, j, h, w, size=(self.img_size, self.img_size))  # fixed line
            if flip:
                img = TF.hflip(img)
            if apply_blur:
                img = self.blur(img)
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.5], std=[0.5])
            processed.append(img)

        return torch.cat(processed, dim=0)  # shape [5, H, W]

class Augment:
    """
    Applies synchronized stochastic augmentation for grayscale multi-view images.
    """

    def __init__(self, img_size, s=1):
        self.train_transform = SynchronizedAugment(img_size, s)

    def __call__(self, imgs):
        """
        imgs: list of 5 PIL images (views)
        Returns:
            Two tensors: x_i and x_j, both of shape [5, 224, 224]
        """
        return self.train_transform(imgs), self.train_transform(imgs)
    
class TestSynchronizedAugment:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, imgs):
        processed = []
        for img in imgs:
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.5], std=[0.5])
            processed.append(img)
        return torch.cat(processed, dim=0)  # shape [5, H, W]

class TestAugment:
    def __init__(self, img_size):
        self.train_transform = TestSynchronizedAugment(img_size)

    def __call__(self, imgs):
        """
        imgs: list of 5 PIL images (views)
        Returns:
            Two tensors: x_i and x_j, both of shape [5, 224, 224]
        """
        return self.train_transform(imgs), self.train_transform(imgs)
    
class MultiViewImageDataset(Dataset):
    def __init__(self, dataframe, base_path, num_views=5, transform=None):
        self.df = dataframe
        self.base_path = base_path
        self.num_views = num_views
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        base_name = self.df.iloc[idx]['location_name_2']
        views = []
        for i in range(self.num_views):
            file_name = f"{base_name}-{i}.ppm"
            full_path = os.path.join(self.base_path, file_name)

            try:
                img = Image.open(full_path).convert('L')
                views.append(img)
            except FileNotFoundError:
                print(f"Missing image: {full_path}")
                views.append(Image.new('L', (224, 224)))

        if self.transform:
            views1, views2 = self.transform(views)
        else:
            temp_views = []
            for view in views:
                temp_view = T.ToTensor()(view)
                temp_views.append(temp_view)
            views1 = torch.cat(temp_views, dim=0)
            views2 = views1
        return views1, views2

def get_data_loader(data_df, batch_size, base_path='Eynsham/Images', transform=None, shuffle=True, num_views=5):
    dataset = MultiViewImageDataset(data_df, base_path=base_path, transform=transform, num_views=num_views)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12)
    return loader

def plot_sample_batch(data_loader, num_examples=4, num_views=5):
    data_iter = iter(data_loader)
    batch, _ = next(data_iter)
    fig, axes = plt.subplots(num_examples, num_views, figsize=(15, 3 * num_examples))

    for i in range(num_examples):
        for j in range(num_views):
            ax = axes[i][j] if num_examples > 1 else axes[j]
            img = batch[i][j]
            ax.imshow(img.numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Example {i}, View {j}')
    plt.tight_layout()
    plt.show()

class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        self.mask = self.mask.to(self.device)
        similarity_matrix = similarity_matrix.to(self.device)
        denominator = self.mask * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss

class AddProjection(nn.Module):
    def __init__(self, embedding_size, mlp_dim=512, use_adapter=True):
        super(AddProjection, self).__init__()
        
        self.use_adapter = use_adapter
        if self.use_adapter:

            self.channel_adapter = nn.Conv2d(
                in_channels=5, 
                out_channels=3, 
                kernel_size=1, 
                stride=1
            )

        self.backbone = models.resnet18(weights=None, num_classes=mlp_dim)
        self.backbone.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        if self.use_adapter:
            x = self.channel_adapter(x)  # Shape: [B, 5, 224, 224] → [B, 3, 224, 224]
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)
    
class AddProjectionParallel(nn.Module):
    def __init__(self, embedding_size, mlp_dim=512, view_size=5):
        super(AddProjectionParallel, self).__init__()

        self.mlp_dim = mlp_dim
        self.backbone = models.resnet18(weights=None, num_classes=mlp_dim)
        self.backbone.fc = nn.Identity()
        self.embedding_concat = nn.Linear(view_size * mlp_dim, mlp_dim)
        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        B, V, H, W = x.shape #Batch, Views, Height, Width
        x = x.view(B * V, 1, H, W) #BatchxViews, Channels, Height, Width
        x = x.repeat(1, 3, 1, 1)
        embedding_all_views = self.backbone(x)
        embedding_all_views = embedding_all_views.view(B, V*self.mlp_dim)
        final_embedding = F.relu(self.embedding_concat(embedding_all_views))
        if return_embedding:
            return final_embedding
        return self.projection(final_embedding)
    
def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups


class SimCLR_pl(pl.LightningModule):
    def __init__(self, embedding_size, mlp_dim, use_adapter=True, parallel_views=False):
        super().__init__()
        if parallel_views:
            self.model = AddProjectionParallel(embedding_size=embedding_size, mlp_dim=mlp_dim)
        else:
            self.model = AddProjection(embedding_size=embedding_size, mlp_dim=mlp_dim, use_adapter=use_adapter)
        self.loss = ContrastiveLoss(BATCH_SIZE, device, temperature=0.5)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch):
        x1, x2 = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)
        self.log('Contrastive loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        max_epochs = MAX_EPOCHS
        param_groups = define_param_groups(self.model, WEIGHT_DECAY, OPTIMIZER)
        lr = LR
        optimizer = Adam(param_groups, lr=lr, weight_decay=WEIGHT_DECAY)

        print(f'Optimizer Adam, '
              f'Learning Rate {lr}, '
              f'Effective batch size {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup]
    
def get_embeddings(model, df, base_path, column_name, transform, batch_size=64, device='cuda', num_views=5):
    dataset = MultiViewImageDataset(df.assign(location_name_2=df[column_name]), base_path=base_path, num_views=num_views, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in loader:
            batch = batch[0]
            batch = batch.to(device)
            emb = model(batch,)
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)

def project_embeddings(embeddings, target_dim=2):
    if embeddings.shape[1] == target_dim:
        return embeddings, None
    pca = PCA(n_components=target_dim)
    return torch.tensor(pca.fit_transform(embeddings), dtype=torch.float)

def plot_embedding_match_2d(model, df, base_path, transform, device='cuda', batch_size=64, use_pca = True, num_views=5):
    emb_2 = get_embeddings(model, df, base_path, 'location_name_2', transform, batch_size, device, num_views=num_views)

    idx = random.randint(0, len(df) - 1)
    sample_df = df.iloc[[idx]]
    emb_1 = get_embeddings(model, sample_df, base_path, 'location_name_1', transform, batch_size=1, device=device, num_views=num_views)
    emb_2_sample = get_embeddings(model, sample_df, base_path, 'location_name_2', transform, batch_size=1, device=device, num_views=num_views)

    if use_pca:
        complete_tensor = torch.cat([emb_2, emb_2_sample, emb_1], dim=0)
        complete_tensor_proj = project_embeddings(complete_tensor)
        emb_2_proj = complete_tensor_proj[:-2]
        emb_2_sample_proj = complete_tensor_proj[-2]
        emb_1_proj = complete_tensor_proj[-1]
    else:
        emb_2_proj = emb_2
        emb_2_sample_proj = emb_2_sample[0]
        emb_1_proj = emb_1[0]

    plt.figure(figsize=(8, 8))
    plt.scatter(emb_2_proj[:, 0], emb_2_proj[:, 1], alpha=0.3, color='blue', label='location_name_2 (all)')
    plt.scatter(emb_2_sample_proj[0], emb_2_sample_proj[1], color='red', label='Selected location_name_2')
    plt.scatter(emb_1_proj[0], emb_1_proj[1], color='green', label='Corresponding location_name_1')
    plt.legend()
    plt.title(f"Embedding Match - Index {idx}")
    plt.grid(True)
    plt.show()


def evaluate_embedding_accuracy(model, df, base_path, transform, batch_size=64, device='cuda', num_views=5):
    emb_2 = get_embeddings(model, df, base_path, 'location_name_2', transform, batch_size, device, num_views=num_views)
    emb_1 = get_embeddings(model, df, base_path, 'location_name_1', transform, batch_size, device, num_views=num_views)

    emb_2 = F.normalize(emb_2, dim=1)
    emb_1 = F.normalize(emb_1, dim=1)

    similarities = emb_1 @ emb_2.T  # cosine similarity matrix: [N, N]

    top1, top5, top10 = 0, 0, 0
    N = len(df)
    distances = []

    for i in range(N):
        top_k = torch.topk(similarities[i], k=10).indices
        top1_index = top_k[0].item()

        if i == top1_index:
            top1 += 1
        if i in top_k[:5]:
            top5 += 1
        if i in top_k:
            top10 += 1

        # Get true and predicted coordinates
        x1, y1 = df.iloc[i]['N_1'], df.iloc[i]['E_1']
        x2, y2 = df.iloc[top1_index]['N_2'], df.iloc[top1_index]['E_2']

        # Euclidean distance (same units as input coordinates)
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(dist)

    mean_distance = np.mean(distances)

    return top1 / N, top5 / N, top10 / N, mean_distance
