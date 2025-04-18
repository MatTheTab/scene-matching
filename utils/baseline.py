import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter
import random
import os

def compare_ORB(img1, img2, nfeatures=500, ratio_thresh=0.75):
    detector = cv2.ORB_create(nfeatures=nfeatures)
    norm_type = cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm_type)

    orb_kp1, orb_des1 = detector.detectAndCompute(img1, None)
    orb_kp2, orb_des2 = detector.detectAndCompute(img2, None)

    if orb_des1 is None or orb_des2 is None:
        return []  # No features found

    matches = matcher.knnMatch(orb_des1, orb_des2, k=2)
    if len(matches) < 2:
        good = []
    else:
        good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return good

def compare_SIFT(img1, img2, ratio_thresh=0.75):
    detector = cv2.SIFT_create()
    norm_type = cv2.NORM_L2
    matcher = cv2.BFMatcher(norm_type)

    sift_kp1, sift_des1 = detector.detectAndCompute(img1, None)
    sift_kp2, sift_des2 = detector.detectAndCompute(img2, None)

    if sift_des1 is None or sift_des2 is None:
        return []

    matches = matcher.knnMatch(sift_des1, sift_des2, k=2)
    if len(matches) < 2:
        good = []
    else:
        good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return good

def get_baseline(data_df, method, **kargs):
    all_locations_1 = data_df["location_name_1"].tolist()
    all_locations_2 = data_df["location_name_2"].tolist()

    results = {}
    for view in range(5):
        print(f"Processing View: {view}")
        result = get_baseline_single_view(all_locations_1, all_locations_2, view, method, **kargs)
        results[view] = result
    return results

def get_baseline_single_view(locations1, locations2, view, method, **kargs):
    similarities = {}

    for location_1 in tqdm(locations1):
        similarities[location_1] = {}
        img_path_1 = f"./Eynsham/Images/{location_1}-{view}.ppm"
        img_1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
        if img_1 is None or not (img_1.dtype == np.uint8) or not (img_1.min() >= 0) or not (img_1.max() <= 255):
            print(f"Warning: Could not load {img_path_1}")
            continue

        for location_2 in locations2:
            img_path_2 = f"./Eynsham/Images/{location_2}-{view}.ppm"
            img_2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
            if img_2 is None or not (img_2.dtype == np.uint8) or not (img_2.min() >= 0) or not (img_2.max() <= 255):
                print(f"Warning: Could not load {img_path_2}")
                continue

            if method == "ORB":
                nfeatures = kargs.get("nfeatures", 500)
                ratio_thresh = kargs.get("ratio_thresh", 0.75)
                good = compare_ORB(img_1, img_2, nfeatures=nfeatures, ratio_thresh=ratio_thresh)
            elif method == "SIFT":
                ratio_thresh = kargs.get("ratio_thresh", 0.75)
                good = compare_SIFT(img_1, img_2, ratio_thresh=ratio_thresh)
            else:
                raise ValueError("Method must be ORB or SIFT")

            similarities[location_1][location_2] = good
    return similarities

def visualize_top_match(data_df, baseline_results, index=0, view=0, method="ORB", nfeatures=500, ratio_thresh=0.75):
    row = data_df.iloc[index]
    location_1 = row["location_name_1"]

    similarity_dict = baseline_results[view].get(location_1, {})
    if not similarity_dict:
        print(f"No similarity data found for {location_1} in view {view}")
        return

    top_match = max(similarity_dict.items(), key=lambda x: len(x[1]))
    location_2 = top_match[0]
    good_matches = top_match[1]

    path_1 = f"./Eynsham/Images/{location_1}-{view}.ppm"
    path_2 = f"./Eynsham/Images/{location_2}-{view}.ppm"
    img_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE)

    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=nfeatures)
    else:
        detector = cv2.SIFT_create()

    kp1, _ = detector.detectAndCompute(img_1, None)
    kp2, _ = detector.detectAndCompute(img_2, None)

    img_matches = cv2.drawMatches(img_1, kp1, img_2, kp2, good_matches[:20], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if img_matches.ndim == 3:
        img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 6))
    plt.imshow(img_matches, cmap='gray')
    plt.title(f"Best match for '{location_1}' is '{location_2}' using {method}")
    plt.axis("off")
    plt.show()

def evaluate_accuracy_single_view(data_df, baseline_results, view=0, top_k=5):
    correct = 0
    topk_correct = 0
    total = len(data_df)

    for i in range(total):
        location_1 = data_df.iloc[i]["location_name_1"]
        correct_location_2 = data_df.iloc[i]["location_name_2"]

        similarity_dict = baseline_results[view].get(location_1, {})
        if not similarity_dict:
            continue

        ranked = sorted(similarity_dict.items(), key=lambda x: len(x[1]), reverse=True)
        top_matches = [loc for loc, _ in ranked]

        if top_matches[0] == correct_location_2:
            correct += 1

        if correct_location_2 in top_matches[:top_k]:
            topk_correct += 1

    acc = correct / total
    topk_acc = topk_correct / total
    return acc, topk_acc

def evaluate_accuracy_freq_aggregation(data_df, baseline_results, top_k=5):
    correct = 0
    topk_correct = 0
    total = len(data_df)

    for i in range(total):
        location_1 = data_df.iloc[i]["location_name_1"]
        correct_location_2 = data_df.iloc[i]["location_name_2"]

        all_ranks = []
        for view in range(5):
            similarity_dict = baseline_results[view].get(location_1, {})
            if not similarity_dict:
                continue

            ranked = sorted(similarity_dict.items(), key=lambda x: len(x[1]), reverse=True)
            top_matches = [loc for loc, _ in ranked]
            for match in top_matches:
                all_ranks.append(match)

        top_frequent = [item for item, count in Counter(all_ranks).most_common()]
        if top_frequent[0] == correct_location_2:
            correct += 1

        if correct_location_2 in top_frequent[:top_k]:
            topk_correct += 1

    acc = correct / total
    topk_acc = topk_correct / total
    return acc, topk_acc

def evaluate_accuracy_weighted_aggregation(data_df, baseline_results, top_k=5):
    correct = 0
    topk_correct = 0
    total = len(data_df)

    for i in range(total):
        location_1 = data_df.iloc[i]["location_name_1"]
        correct_location_2 = data_df.iloc[i]["location_name_2"]

        all_ranks = []
        for view in range(5):
            similarity_dict = baseline_results[view].get(location_1, {})
            if not similarity_dict:
                continue

            ranked = sorted(similarity_dict.items(), key=lambda x: len(x[1]), reverse=True)
            top_matches = [(loc, len(matches)) for loc, matches in ranked]
            for match in top_matches:
                all_ranks.append(match)

        unique_images = set([])
        top_weighted = []
        top_scores = []
        for img_name, score in all_ranks:
            if img_name in unique_images:
                top_scores[top_weighted.index(img_name)] += score
            else:
                unique_images.add(img_name)
                top_weighted.append(img_name)
                top_scores.append(score)

        top_scores, top_weighted = zip(*sorted(zip(top_scores, top_weighted), reverse=True))
        top_weighted = list(top_weighted)
        
        if top_weighted[0] == correct_location_2:
            correct += 1
        if correct_location_2 in top_weighted[:top_k]:
            topk_correct += 1

    acc = correct / total
    topk_acc = topk_correct / total
    return acc, topk_acc

