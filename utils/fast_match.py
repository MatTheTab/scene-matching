from utils.baseline import *
from skimage.measure import shannon_entropy
from scipy.ndimage import center_of_mass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torchvision import models, transforms
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import umap
from utils.model_utils import load_model
from sklearn.metrics.pairwise import cosine_similarity

def plot_clusters(data_df, locations_encoded, all_labels):
    import matplotlib.pyplot as plt

    for view in range(5):
        features = np.array(locations_encoded[view])
        labels = np.array(all_labels[view])

        # Get corresponding locations for this view
        loc1_names = data_df["location_name_1"].tolist()
        loc2_names = data_df["location_name_2"].tolist()

        # Map names to their descriptors for current view
        loc1_desc = []
        loc1_labels = []
        loc2_desc = []
        loc2_labels = []

        for i in range(len(data_df)):
            loc1_desc.append(locations_encoded[view][i])
            loc2_desc.append(locations_encoded[view][i])
            loc1_labels.append(all_labels[view][i])
            loc2_labels.append(all_labels[view][i])

        loc1_desc = np.array(loc1_desc)
        loc2_desc = np.array(loc2_desc)
        loc1_labels = np.array(loc1_labels)
        loc2_labels = np.array(loc2_labels)

        reducer_umap = umap.UMAP(n_components=2, random_state=42)
        reducer_tsne = TSNE(n_components=2, random_state=42)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"View {view} - 2D Projections of Descriptors", fontsize=16)

        # Location 1: UMAP
        loc1_umap = reducer_umap.fit_transform(loc1_desc)
        axes[0, 0].scatter(loc1_umap[:, 0], loc1_umap[:, 1], c=loc1_labels, cmap='tab10', s=10)
        axes[0, 0].set_title("Location 1 - UMAP")

        # Location 1: t-SNE
        loc1_tsne = reducer_tsne.fit_transform(loc1_desc)
        axes[0, 1].scatter(loc1_tsne[:, 0], loc1_tsne[:, 1], c=loc1_labels, cmap='tab10', s=10)
        axes[0, 1].set_title("Location 1 - t-SNE")

        # Location 2: UMAP
        loc2_umap = reducer_umap.fit_transform(loc2_desc)
        axes[1, 0].scatter(loc2_umap[:, 0], loc2_umap[:, 1], c=loc2_labels, cmap='tab10', s=10)
        axes[1, 0].set_title("Location 2 - UMAP")

        # Location 2: t-SNE
        loc2_tsne = reducer_tsne.fit_transform(loc2_desc)
        axes[1, 1].scatter(loc2_tsne[:, 0], loc2_tsne[:, 1], c=loc2_labels, cmap='tab10', s=10)
        axes[1, 1].set_title("Location 2 - t-SNE")

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Histogram plot (Second figure)
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle(f"View {view} - Cluster Label Distributions", fontsize=16)

        sns.histplot(loc1_labels, bins=len(np.unique(loc1_labels)), discrete=True, ax=ax1, palette='tab10')
        ax1.set_title("Location 1 Cluster Histogram")
        ax1.set_xlabel("Cluster Label")
        ax1.set_ylabel("Count")

        sns.histplot(loc2_labels, bins=len(np.unique(loc2_labels)), discrete=True, ax=ax2, palette='tab10')
        ax2.set_title("Location 2 Cluster Histogram")
        ax2.set_xlabel("Cluster Label")
        ax2.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("OpenIBL", gpu=(device == "cuda"))
model.eval()

transformer = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.458, 0.408],
        std=[1/255] * 3
    )
])


def load_image_tensor(location, view):
    path = f"./Eynsham/Images/{location}-{view}.ppm"
    img = Image.open(path).convert('RGB')
    tensor = transformer(img).unsqueeze(0).to(device)
    return tensor

def get_deep_description(location, view):
    with torch.no_grad():
        embedding = model(load_image_tensor(location, view))[0].cpu().numpy()
    return embedding

def get_classical_description(location, view):
    path = f"./Eynsham/Images/{location}-{view}.ppm"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    entropy = shannon_entropy(img)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    com = center_of_mass(img)
    return np.concatenate([[entropy], hu_moments, [edge_density], com])

def cluster(features, n_clusters=10):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled)
    return kmeans, scaler, labels

def match_descriptors(data_df, method, descriptors, all_clusters, all_scalers, all_labels):
    results = {}
    for view in range(5):
        clusterer, scaler, labels = all_clusters[view], all_scalers[view], all_labels[view]
        similarities, desc_loc1, desc_loc2, loc1_labels = {}, {}, {}, {}

        # Precompute descriptors and clusters for loc1
        for idx, row in data_df.iterrows():
            loc1 = row["location_name_1"]
            img_path = f"./Eynsham/Images/{loc1}-{view}.ppm"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.dtype != np.uint8:
                print(f"Warning: Cannot load {img_path}")
                continue

            des1 = get_ORB(img, 500)[1] if method == "ORB" else get_SIFT(img)[1]
            desc_loc1[loc1] = des1

            features = get_classical_description(loc1, view) if descriptors == "classical" else get_deep_description(loc1, view)
            pred_cluster = clusterer.predict(scaler.transform([features]))[0]
            loc1_labels[loc1] = pred_cluster

        # Precompute descriptors for loc2
        for idx, row in data_df.iterrows():
            loc2 = row["location_name_2"]
            img_path = f"./Eynsham/Images/{loc2}-{view}.ppm"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.dtype != np.uint8:
                print(f"Warning: Cannot load {img_path}")
                continue

            des2 = get_ORB(img, 500)[1] if method == "ORB" else get_SIFT(img)[1]
            desc_loc2[loc2] = des2

        # Compute similarities
        for i, row in data_df.iterrows():
            loc1 = row["location_name_1"]
            loc1_label = loc1_labels.get(loc1)
            des1 = desc_loc1.get(loc1)
            similarities[loc1] = {}

            for j, row2 in data_df.iterrows():
                loc2 = row2["location_name_2"]
                des2 = desc_loc2.get(loc2)
                if labels[j] == loc1_label:
                    matches = compare_ORB(des1, des2) if method == "ORB" else compare_SIFT(des1, des2)
                else:
                    matches = []
                similarities[loc1][loc2] = matches

        results[view] = similarities
    return results

def compute_vlad_similarity(data_df, descriptors, all_clusters, all_scalers, all_labels, locations_encoded):
    results = {}
    for view in range(5):
        clusterer, scaler, labels = all_clusters[view], all_scalers[view], all_labels[view]
        similarities, loc1_embeddings, loc2_embeddings, loc1_labels = {}, {}, {}, {}

        for i, row in data_df.iterrows():
            loc1, loc2 = row["location_name_1"], row["location_name_2"]
            loc2_embeddings[loc2] = locations_encoded[view][i]
            desc = get_classical_description(loc1, view) if descriptors == "classical" else get_deep_description(loc1, view)
            loc1_embeddings[loc1] = get_deep_description(loc1, view)
            pred_cluster = clusterer.predict(scaler.transform([desc]))[0]
            loc1_labels[loc1] = pred_cluster

        for i, row in data_df.iterrows():
            loc1 = row["location_name_1"]
            similarities[loc1] = {}
            vec1 = loc1_embeddings[loc1].reshape(1, -1)
            label1 = loc1_labels[loc1]

            for j, row2 in data_df.iterrows():
                loc2 = row2["location_name_2"]
                vec2 = loc2_embeddings[loc2].reshape(1, -1)
                sim = cosine_similarity(vec1, vec2)[0][0] if labels[j] == label1 else 0.0
                similarities[loc1][loc2] = sim

        results[view] = similarities
    return results

def find_images_fast(data_df, descriptors, method, plot=False):
    locations_encoded = {view: [
        get_classical_description(row["location_name_2"], view) if descriptors == "classical"
        else get_deep_description(row["location_name_2"], view)
        for _, row in data_df.iterrows()] for view in range(5)}

    all_clusters, all_scalers, all_labels = {}, {}, {}
    for view in range(5):
        clus, scale, labs = cluster(locations_encoded[view])
        all_clusters[view], all_scalers[view], all_labels[view] = clus, scale, labs

    if plot:
        plot_clusters(data_df, locations_encoded, all_labels)

    if method in ["ORB", "SIFT"]:
        return match_descriptors(data_df, method, descriptors, all_clusters, all_scalers, all_labels)
    elif method == "VLAD":
        return compute_vlad_similarity(data_df, descriptors, all_clusters, all_scalers, all_labels, locations_encoded)
    else:
        raise ValueError("Invalid method specified.")