import os
import re

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from collections import Counter
from collections import defaultdict

import torch
from torchvision import transforms

from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity

def load_model(model_name, gpu=True):
    models = ["OpenIBL"]
    assert model_name in models, f"Available models are: {models}"

    model = None
    if model_name == "OpenIBL":
        model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True).eval()
    
    if model is not None and gpu:
        model = model.cuda()

    return model

def extract_location_embeddings_OpenIBL(df, model, image_folder='Eynsham/Images'):
    # Prepare transform
    transformer = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
            std=[0.00392156862745098] * 3
        )
    ])
        
    # Extract unique location names
    unique_locations = pd.unique(df[['location_name_1', 'location_name_2']].values.ravel())
    
    records = []

    for location in tqdm(unique_locations, desc="Processing Locations"):
        for view in range(5):
            image_filename = f"{location}-{view}.ppm"
            image_path = os.path.join(image_folder, image_filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            try:
                # Load and transform image
                pil_img = Image.open(image_path).convert('RGB')
                img_tensor = transformer(pil_img).unsqueeze(0).cuda()

                # Get embedding
                with torch.no_grad():
                    embedding = model(img_tensor)[0].cpu().numpy()

                records.append({
                    'location_name': location,
                    'view': view,
                    'embedding': embedding
                })
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return pd.DataFrame(records)

def display_gt_pair(match_df, match_df_row_index, view=0, img_dir="Eynsham/Images/"):
    row = match_df.iloc[match_df_row_index]
    img_1_name, img_2_name = row["location_name_1"], row["location_name_2"]
    img_1_path = img_dir + img_1_name + f"-{view}.ppm"
    img_2_path = img_dir + img_2_name + f"-{view}.ppm"

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(img_1_path), cmap="gray")
    axs[0].set_title(f"Query: {img_1_name}")
    axs[0].axis('off')

    axs[1].imshow(Image.open(img_2_path), cmap="gray")
    axs[1].set_title(f"GT Match: {img_2_name}")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def search_closest(location_name_1, view, embeddings_df, image_folder='Eynsham/Images',
                   search_pool='location_name_2', base_df=None, show=True):
    """
    Search the closest match for a given location_name_1 in a specific view.
    Optionally display the query and top-1 result side by side.
    """
    # Filter for the given view
    view_embeddings = embeddings_df[embeddings_df['view'] == view]
    emb_dict = dict(zip(view_embeddings['location_name'], view_embeddings['embedding']))
    
    if location_name_1 not in emb_dict:
        raise ValueError(f"{location_name_1} not found in embeddings for view {view}")
    
    emb1 = emb_dict[location_name_1]

    # Build the candidate list
    if search_pool == 'location_name_2' and base_df is not None:
        candidates = base_df['location_name_2'].unique()
    else:
        candidates = view_embeddings['location_name'].unique()

    candidates = [loc for loc in candidates if loc in emb_dict and loc != location_name_1]
    emb2_list = [emb_dict[loc] for loc in candidates]

    if not emb2_list:
        raise ValueError("No valid comparison candidates found.")

    # Calculate cosine distances and get ranked list
    dists = cosine_distances([emb1], emb2_list)[0]
    ranked = sorted(zip(candidates, dists), key=lambda x: x[1])

    top1_match = ranked[0][0]

    # Show query and result images
    if show:
        query_path = os.path.join(image_folder, f"{location_name_1}-{view}.ppm")
        result_path = os.path.join(image_folder, f"{top1_match}-{view}.ppm")
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(Image.open(query_path), cmap="gray")
        axs[0].set_title(f"Query: {location_name_1}")
        axs[0].axis('off')

        axs[1].imshow(Image.open(result_path), cmap="gray")
        axs[1].set_title(f"Top-1 Match: {top1_match}")
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    return ranked

def evaluate_accuracy_per_view(base_df, embeddings_df):
    results = {}

    for view in range(5):
        # Filter embeddings by view
        view_embeddings = embeddings_df[embeddings_df['view'] == view]
        emb_dict = dict(zip(view_embeddings['location_name'], view_embeddings['embedding']))
        
        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        total = 0

        print(f"Calculating results for view {view}")
        for _, row in tqdm(base_df.iterrows()):
            loc1 = row['location_name_1']
            loc2_true = row['location_name_2']
            
            if loc1 not in emb_dict:
                continue
            
            emb1 = emb_dict[loc1]
            loc2_candidates = [loc for loc in base_df['location_name_2'].unique() if loc in emb_dict]
            emb2_list = [emb_dict[loc2] for loc2 in loc2_candidates]
            
            if not emb2_list:
                continue
            
            dists = cosine_distances([emb1], emb2_list)[0]
            sorted_indices = np.argsort(dists)
            sorted_locs = [loc2_candidates[i] for i in sorted_indices]

            # Evaluate Top-1, Top-5, Top-10
            if loc2_true == sorted_locs[0]:
                top1_correct += 1
            if loc2_true in sorted_locs[:5]:
                top5_correct += 1
            if loc2_true in sorted_locs[:10]:
                top10_correct += 1
            
            total += 1

        results[view] = {
            'top1_accuracy': top1_correct / total if total > 0 else 0,
            'top5_accuracy': top5_correct / total if total > 0 else 0,
            'top10_accuracy': top10_correct / total if total > 0 else 0,
            'count': total
        }
    
    return results


def evaluate_accuracy_majority_vote(base_df, embeddings_df, top_ns=[1, 5, 10]):
    # Create view-specific embedding dicts
    emb_by_view = {
        view: dict(zip(df_view['location_name'], df_view['embedding']))
        for view, df_view in embeddings_df.groupby('view')
    }

    correct_top = {n: 0 for n in top_ns}
    total = 0

    print(f"Calculating results")
    for _, row in tqdm(base_df.iterrows()):
        loc1 = row['location_name_1']
        loc2_true = row['location_name_2']
        votes = []

        for view in range(5):
            emb_dict = emb_by_view.get(view, {})
            if loc1 not in emb_dict:
                continue

            emb1 = emb_dict[loc1]
            loc2_candidates = [loc for loc in base_df['location_name_2'].unique() if loc in emb_dict]
            emb2_list = [emb_dict[loc2] for loc2 in loc2_candidates]
            
            if not emb2_list:
                continue

            dists = cosine_distances([emb1], emb2_list)[0]
            sorted_indices = np.argsort(dists)
            ranked_locs = [loc2_candidates[i] for i in sorted_indices]

            votes.extend(ranked_locs[:max(top_ns)])  # gather top-N votes per view

        if not votes:
            continue

        total += 1
        vote_counts = Counter(votes)
        ranked_vote_list = [loc for loc, _ in vote_counts.most_common()]

        for n in top_ns:
            top_n_preds = ranked_vote_list[:n]
            if loc2_true in top_n_preds:
                correct_top[n] += 1

    return {
        f'top{n}_accuracy': correct_top[n] / total if total > 0 else 0
        for n in top_ns
    } | {'count': total}


def save_embeddings_to_npy(embeddings_df, output_folder='embeddings_npy'):
    """
    Saves each embedding as a separate .npy file in the format <location_name>-<view>.npy
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for _, row in embeddings_df.iterrows():
        location = row['location_name']
        view = row['view']
        embedding = row['embedding']
        
        file_name = f"{location}-{view}.npy"
        path = os.path.join(output_folder, file_name)
        np.save(path, embedding)

def load_embeddings_from_npy_folder(folder='embeddings_npy'):
    """
    Loads all .npy files in a folder into a DataFrame with columns:
    - 'location_name'
    - 'view'
    - 'embedding'
    """
    data = []
    pattern = re.compile(r'^(.*)-(\d+)\.npy$')  # matches <location_name>-<view>.npy

    print("Loading embeddings...")
    for file in tqdm(os.listdir(folder)):
        if file.endswith('.npy'):
            match = pattern.match(file)
            if match:
                location_name = match.group(1)
                view = int(match.group(2))
                embedding = np.load(os.path.join(folder, file))
                data.append({'location_name': location_name, 'view': view, 'embedding': embedding})

    return pd.DataFrame(data)

def extract_concatenate_embeddings_OpenIBL(df, model, image_folder='Eynsham/Images'):
    # Prepare transform
    transformer = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
            std=[0.00392156862745098] * 3
        )
    ])
        
    # Extract unique location names
    unique_locations = pd.unique(df[['location_name_1', 'location_name_2']].values.ravel())
    
    locations_embeddings = {}

    for location in tqdm(unique_locations, desc="Processing Locations"):
        concat_embedding = np.array([])
        for view in range(5):
            image_filename = f"{location}-{view}.ppm"
            image_path = os.path.join(image_folder, image_filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            try:
                # Load and transform image
                pil_img = Image.open(image_path).convert('RGB')
                img_tensor = transformer(pil_img).unsqueeze(0).cuda()

                # Get embedding
                with torch.no_grad():
                    embedding = model(img_tensor)[0].cpu().numpy()
                concat_embedding = np.concatenate((concat_embedding, embedding), axis=0)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        locations_embeddings[location] = concat_embedding.copy()
    return locations_embeddings

def concat_evaluate_matching_accuracy(data_df, location_embeddings):
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    all_location_2_names = data_df['location_name_2'].unique()
    all_location_2_embeddings = {loc: location_embeddings[loc] for loc in all_location_2_names}

    for i in range(len(data_df)):
        loc1 = data_df.iloc[i]['location_name_1']
        loc2_correct = data_df.iloc[i]['location_name_2']

        if loc1 not in location_embeddings or loc2_correct not in location_embeddings:
            continue

        emb1 = location_embeddings[loc1].reshape(1, -1)  # (1, D)
        
        similarities = []
        for loc2 in all_location_2_names:
            emb2 = location_embeddings.get(loc2)
            if emb2 is None:
                continue
            emb2 = emb2.reshape(1, -1)
            sim = cosine_similarity(emb1, emb2)[0][0]
            similarities.append((loc2, sim))

        # Sort by similarity, descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Extract top N
        top_matches = [loc for loc, _ in similarities]

        if loc2_correct == top_matches[0]:
            top1_correct += 1
        if loc2_correct in top_matches[:5]:
            top5_correct += 1
        if loc2_correct in top_matches[:10]:
            top10_correct += 1

    total = len(data_df)

    return top1_correct / total, top5_correct / total, top10_correct / total
