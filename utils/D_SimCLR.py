from utils.SimCLR import *

def get_positive_pairs(similarity_matrix, batch_size):
    device = similarity_matrix.device
    indices = torch.arange(batch_size).to(device)
    sim_ij = similarity_matrix[indices, indices + batch_size]
    sim_ji = similarity_matrix[indices + batch_size, indices]
    return torch.cat([sim_ij, sim_ji], dim=0)

class InverseDistanceContrastiveLoss(ContrastiveLoss):
    def __init__(self, batch_size, device, temperature=0.5, epsilon=1e-8):
        super().__init__(batch_size, device, temperature)
        self.epsilon = epsilon

    def forward(self, proj_1, proj_2, distances):
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)
        batch_size = proj_1.shape[0]

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        positive_distances_ij = torch.diag(distances, batch_size)
        positive_distances_ji = torch.diag(distances, -batch_size)
        positive_distances = torch.cat([positive_distances_ij, positive_distances_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        self.mask = (~torch.eye(batch_size * 2, dtype=bool)).float().to(self.device)
        similarity_matrix = similarity_matrix.to(self.device)
        denominator = self.mask * torch.exp(similarity_matrix / self.temperature)

        weights = 1 / (positive_distances + self.epsilon)
        weights = weights.to(self.device)
        
        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        weighted_loss = (weights * all_losses).sum() / (2 * batch_size)
        return weighted_loss
    
class SoftDenominatorContrastiveLoss(ContrastiveLoss):
    def __init__(self, batch_size, device, temperature=0.5, lambda_=1.0, epsilon=1e-8):
        super().__init__(batch_size, device, temperature)
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def forward(self, proj_1, proj_2, distances):
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)
        batch_size = proj_1.shape[0]

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        self.mask = (~torch.eye(batch_size * 2, dtype=bool)).float().to(self.device)

        adjusted_sim = similarity_matrix - self.lambda_ * distances.to(self.device)
        denominator = self.mask * torch.exp(adjusted_sim / self.temperature)
        denominator += self.epsilon

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss
    
class TemperatureScalingContrastiveLoss(ContrastiveLoss):
    def __init__(self, batch_size, device, temperature=0.5, epsilon=1e-8):
        super().__init__(batch_size, device, temperature)
        self.epsilon = epsilon

    def forward(self, proj_1, proj_2, distances):
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)
        batch_size = proj_1.shape[0]

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        dist_ij = torch.diag(distances, batch_size)
        dist_ji = torch.diag(distances, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        positives_distances = torch.cat([dist_ij, dist_ji], dim=0)
        num_scaled_temperature = positives_distances + self.temperature
        denom_scaled_temperature = distances + self.temperature

        nominator = torch.exp(positives / num_scaled_temperature)
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        self.mask = self.mask.to(self.device)
        similarity_matrix = similarity_matrix.to(self.device)
        denominator = self.mask * torch.exp(similarity_matrix / denom_scaled_temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss
    
class DistanceMultiViewImageDataset(Dataset):
    def __init__(self, dataframe, base_path, num_views=5, transform=None):
        self.df = dataframe
        self.base_path = base_path
        self.num_views = num_views
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        loc_1_name = self.df.iloc[idx]['location_name_1']
        loc_2_name = self.df.iloc[idx]['location_name_2']
        loc1_n = self.df.iloc[idx]['N_1']
        loc1_e = self.df.iloc[idx]['E_1']
        loc2_n = self.df.iloc[idx]['N_2']
        loc2_e = self.df.iloc[idx]['E_2']
        views_1 = []
        views_2 = []
        for i in range(self.num_views):
            file_name_1 = f"{loc_1_name}-{i}.ppm"
            full_path_1 = os.path.join(self.base_path, file_name_1)
            file_name_2 = f"{loc_2_name}-{i}.ppm"
            full_path_2 = os.path.join(self.base_path, file_name_2)

            try:
                img_1 = Image.open(full_path_1).convert('L')
                views_1.append(img_1)
            except FileNotFoundError:
                print(f"Missing image: {full_path_1}")
                views_1.append(Image.new('L', (224, 224)))

            try:
                img_2 = Image.open(full_path_2).convert('L')
                views_2.append(img_2)
            except FileNotFoundError:
                print(f"Missing image: {full_path_2}")
                views_2.append(Image.new('L', (224, 224)))

        if self.transform:
            views1, _ = self.transform(views_1)
            views2, _ = self.transform(views_2)

        else:
            temp_views_1 = []
            temp_views_2 = []
            for view in views_1:
                temp_view = T.ToTensor()(view)
                temp_views_1.append(temp_view)
            views1 = torch.cat(temp_views_1, dim=0)

            for view in views_2:
                temp_view = T.ToTensor()(view)
                temp_views_2.append(temp_view)
            views2 = torch.cat(temp_views_2, dim=0)
        return views1, views2, loc1_n, loc1_e, loc2_n, loc2_e
    
def get_distance_loader(data_df, batch_size, base_path='Eynsham/Images', transform=None, shuffle=True, num_views=5):
    dataset = DistanceMultiViewImageDataset(data_df, base_path=base_path, transform=transform, num_views=num_views)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=12)
    return loader

def plot_sample_distance_batch(data_loader, num_examples=4, num_views=3):
    data_iter = iter(data_loader)
    batch1, batch2, loc1_n, loc1_e, loc2_n, loc2_e = next(data_iter)

    fig, axes = plt.subplots(num_examples * 2, num_views, figsize=(3 * num_views, 3 * num_examples * 2))

    for i in range(num_examples):
        for view in range(num_views):
            # Plot from batch1
            img1 = batch1[i][view].unsqueeze(0)  # [1, H, W]
            ax1 = axes[i * 2][view]
            ax1.imshow(TF.to_pil_image(img1), cmap='gray')
            ax1.axis('off')
            if view == 0:
                ax1.set_title(f"Batch 1\nN={loc1_n[i]:.3f}, E={loc1_e[i]:.3f}")

            # Plot from batch2
            img2 = batch2[i][view].unsqueeze(0)  # [1, H, W]
            ax2 = axes[i * 2 + 1][view]
            ax2.imshow(TF.to_pil_image(img2), cmap='gray')
            ax2.axis('off')
            if view == 0:
                ax2.set_title(f"Batch 2\nN={loc2_n[i]:.3f}, E={loc2_e[i]:.3f}")

    plt.tight_layout()
    plt.show()


def get_embeddings_from_distance_loader(model, data_loader, device='cuda'):
    all_embeddings = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in data_loader:
            x1, _, _, _, _, _ = batch  # only x1 is used for embeddings here
            x1 = x1.to(device)
            emb = model(x1)
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)

def project_embeddings_distance(embeddings, target_dim=2):
    if embeddings.shape[1] == target_dim:
        return embeddings, None
    pca = PCA(n_components=target_dim)
    return torch.tensor(pca.fit_transform(embeddings.detach().numpy()), dtype=torch.float)

def plot_embedding_match_2d_distance_loader(model, df, base_path, transform, device='cuda', batch_size=64, use_pca=True, num_views=5):
    # Get full dataset embeddings
    loader_all = get_distance_loader(df, batch_size, base_path=base_path, transform=transform, shuffle=False, num_views=num_views)
    emb_2 = get_embeddings_from_distance_loader(model, loader_all, device=device)

    # Pick one sample
    idx = random.randint(0, len(df) - 1)
    sample_df = df.iloc[[idx]]
    loader_sample = get_distance_loader(sample_df, 1, base_path=base_path, transform=transform, shuffle=False, num_views=num_views)
    sample_batch = next(iter(loader_sample))
    x1_sample, x2_sample, *_ = sample_batch

    x1_sample = x1_sample.to(device)
    x2_sample = x2_sample.to(device)

    emb_1 = model(x1_sample).cpu()
    emb_2_sample = model(x2_sample).cpu()



    if use_pca:
        complete_tensor = torch.cat([emb_2, emb_2_sample, emb_1], dim=0)
        complete_tensor_proj = project_embeddings_distance(complete_tensor)
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

def evaluate_embedding_accuracy_distance_loader(model, df, base_path, transform, batch_size=64, device='cuda', num_views=5):
    loader = get_distance_loader(df, batch_size, base_path=base_path, transform=transform, shuffle=False, num_views=num_views)
    
    emb_1_list, emb_2_list = [], []
    coords_1, coords_2 = [], []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in loader:
            x1, x2, n1, e1, n2, e2 = batch
            x1 = x1.to(device)
            x2 = x2.to(device)

            emb_1_list.append(model(x1).cpu())
            emb_2_list.append(model(x2).cpu())

            coords_1.extend(zip(n1.tolist(), e1.tolist()))
            coords_2.extend(zip(n2.tolist(), e2.tolist()))

    emb_1 = F.normalize(torch.cat(emb_1_list), dim=1)
    emb_2 = F.normalize(torch.cat(emb_2_list), dim=1)
    similarities = emb_1 @ emb_2.T  # [N, N]

    top1, top5, top10 = 0, 0, 0
    distances = []

    N = len(df)
    for i in range(N):
        top_k = torch.topk(similarities[i], k=10).indices
        top1_index = top_k[0].item()

        if i == top1_index:
            top1 += 1
        if i in top_k[:5]:
            top5 += 1
        if i in top_k:
            top10 += 1

        # Euclidean distance between matched coordinates
        x1, y1 = coords_1[i]
        x2, y2 = coords_2[top1_index]
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(dist)

    return top1 / N, top5 / N, top10 / N, np.mean(distances)
