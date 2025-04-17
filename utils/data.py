import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def display_views(base_path='Eynsham/Images', base_name='grab_0.000', num_views=5):
    """
    Display multiple viewpoints of a .ppm image series.
    
    Parameters:
        base_path (str): Path to the folder containing the images.
        base_name (str): Base name of the image series, e.g., 'grab_0.000'.
        num_views (int): Number of views to load and display (default is 5).
    """
    fig, axes = plt.subplots(1, num_views, figsize=(15, 4))
    for i in range(num_views):
        file_name = f"{base_name}-{i}.ppm"
        full_path = os.path.join(base_path, file_name)
        try:
            img = Image.open(full_path)
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f'View {i}')
            axes[i].axis('off')
        except FileNotFoundError:
            axes[i].set_title(f'Missing: {i}')
            axes[i].axis('off')
            print(f"File not found: {full_path}")
    plt.tight_layout()
    plt.show()

def extract_timestamp(filename):
    """
    Extracts the numeric timestamp from the filename.

    This function takes a filename in the format 'grab_1234567890.123-0.ppm', extracts the numeric timestamp
    (e.g., '1234567890.123') from the filename and returns it as a float.

    Parameters:
        filename (str): The name of the file to extract the timestamp from, in the format 'grab_TIMESTAMP-N.view.ppm'.
        
    Returns:
        float: The extracted timestamp value if found, else None.
    """
    
    match = re.search(r'grab_(\d+\.\d+)-\d+\.ppm', filename)
    return float(match.group(1)) if match else None

def count_loop_pair(base_dir='Eynsham/Images', loop_closure='grab_1216907830.929-0.ppm', num_views=5):
    """
    Count the number of photos before and after a specified loop closure point.

    This function processes the image files in a given directory, identifies the loop closure point, 
    and calculates how many images are available before and after that loop closure. It also calculates
    the difference in the number of photos in both segments.

    Parameters:
        base_dir (str): Directory containing the image files. Default is 'Eynsham/Images'.
        loop_closure (str): The filename representing the loop closure image (e.g., 'grab_1216907830.929-0.ppm').
        num_views (int): The number of views to display (not used directly in this function but available for future use).

    Prints:
        Total length of data
        Number of photos before the loop closure
        Number of photos after the loop closure
        The difference in the number of photos before and after the loop closure
        
    Example:
        count_loop_pair(base_dir='path/to/images', loop_closure='grab_1216907830.929-0.ppm', num_views=5)
    """

    all_files = [f for f in os.listdir(base_dir) if f.endswith('.ppm')]
    base_names = sorted(list(set(
        re.match(r'(grab_\d+\.\d+)-\d+\.ppm', f).group(1)
        for f in all_files
        if re.match(r'(grab_\d+\.\d+)-\d+\.ppm', f)
    )), key=lambda name: extract_timestamp(name + '-0.ppm'))


    loop_base = re.match(r'(grab_\d+\.\d+)-\d+\.ppm', loop_closure).group(1)
    loop_idx = base_names.index(loop_base)
    offset = len(base_names[:loop_idx+1]) - len(base_names[loop_idx:])
    print(f"Total length of data: {len(base_names)}")
    print(f"Photos before loop: {len(base_names[:loop_idx+1])}")
    print(f"Photos after loop: {len(base_names[loop_idx:])}")
    print(f"Difference in number of photos: {offset}")

def parse_log_with_gps_alignment(filepath, first_match_override=True):
    """
    Parse a log file to extract GPS and image data, aligning them based on the last recorded GPS entry.

    This function processes the log file line by line. It captures GPS data (time, N, E, X, Y) when it finds a GPS line
    and associates the closest GPS data with subsequent LADYBUG_GRAB entries. The result is a list of dictionaries where
    each entry contains the base name of the image and the corresponding GPS coordinates (N, E, X, Y).

    Parameters:
        filepath (str): The path to the log file to parse.
        first_match_override (bool): If True, overrides the first match with the next one. Default is True.
        
    Returns:
        list: A list of dictionaries, each containing the base image name and corresponding GPS data (N, E, X, Y).
    
    Example:
        entries = parse_log_with_gps_alignment('path/to/log_file.txt')
    """
    entries = []
    last_gps = None

    with open(filepath, 'r') as file:
        header_skipped = False

        for line in file:
            line = line.strip()

            # Skip header
            if not header_skipped:
                if not line.startswith('%'):
                    header_skipped = True
                else:
                    continue

            # --------- GPS Line ---------
            if 'GPS' in line and 'iGPS' in line:
                match = re.search(
                    r'time=([\d.]+),N=([\d.\-]+),E=([\d.\-]+),X=([\d.\-]+),Y=([\d.\-]+)', line
                )
                if match:
                    gps_time, N, E, X, Y = match.groups()
                    last_gps = {
                        'gps_time': float(gps_time),
                        'N': float(N),
                        'E': float(E),
                        'X': float(X),
                        'Y': float(Y)
                    }

            # --------- LADYBUG_GRAB Line ---------
            elif 'LADYBUG_GRAB' in line:
                file_match = re.search(r'File0=.*?/(grab_\d+\.\d+)-0\.ppm', line)
                if file_match and last_gps:
                    base_name = file_match.group(1)
                    entry = {
                        'base_name': base_name,
                        'N': last_gps['N'],
                        'E': last_gps['E'],
                        'X': last_gps['X'],
                        'Y': last_gps['Y']
                    }
                    entries.append(entry)

    return entries

def find_closest_entry(source_row, target_df):
    """
    Finds the closest entry in `target_df` to `source_row` using Euclidean distance (N, E).

    This function calculates the Euclidean distance between the GPS coordinates of a source row (N, E) from `df1` 
    and all entries in the target dataframe (`df2`). The function returns the row from `df2` that is closest to 
    the source row based on the smallest Euclidean distance between the N and E coordinates.

    Parameters:
        source_row (pandas.Series): A single row from the source dataframe containing GPS coordinates (N, E).
        target_df (pandas.DataFrame): The dataframe containing the target entries to compare with.

    Returns:
        pandas.Series: The row from `target_df` that is closest to the `source_row` based on Euclidean distance.
    
    Example:
        closest_row = find_closest_entry(source_row, target_df)
    """
    
    dx = target_df['N'] - source_row['N']
    dy = target_df['E'] - source_row['E']
    distances = np.sqrt(dx**2 + dy**2)
    return target_df.iloc[distances.idxmin()]

def create_closest_mapping(df1, df2):
    """
    Create a mapping of the closest entries between two dataframes (`df1` and `df2`) based on their GPS coordinates.

    This function iterates through the rows of the first dataframe (`df1`) and finds the closest matching entry in
    the second dataframe (`df2`) based on the Euclidean distance between their N and E coordinates. It returns a list
    of dictionaries, where each dictionary contains the matching base names and GPS coordinates from both dataframes.

    Parameters:
        df1 (pandas.DataFrame): The first dataframe containing GPS coordinates and image names.
        df2 (pandas.DataFrame): The second dataframe to compare against.
        
    Returns:
        list: A list of dictionaries, each containing matching entries from both dataframes.
    
    Example:
        results = create_closest_mapping(df1, df2)
    """

    results = []
    for i in range(len(df1)):
        row1 = df1.iloc[i]
        row2 = find_closest_entry(row1, df2)
        results_row = {"location_name_1": row1["base_name"],
                    "N_1" : row1["N"],
                    "E_1" : row1["E"],
                    "X_1" : row1["X"],
                    "Y_1" : row1["Y"],
                    "location_name_2": row2["base_name"],
                    "N_2" : row2["N"],
                    "E_2" : row2["E"],
                    "X_2" : row2["X"],
                    "Y_2" : row2["Y"]}
        results.append(results_row.copy())
    return results

def display_dual_views(combined_df, index, base_path='Eynsham/Images', num_views=5):
    """
    Display dual images side-by-side for comparison from two dataframes.

    This function displays images from two different dataframes at the same index for comparison. It shows the images
    in two rows, where the first row corresponds to the image from the first dataframe and the second row corresponds 
    to the closest image from the second dataframe. Each image is displayed in grayscale.

    Parameters:
        combined_df (pandas.DataFrame): A dataframe containing the closest matched entries from two dataframes.
        index (int): The index of the row in `combined_df` to compare the images.
        base_path (str): The base directory where the image files are located. Default is 'Eynsham/Images'.
        num_views (int): Number of views to load and display. Default is 5.
        
    Example:
        display_dual_views(combined_df, index=0, base_path='path/to/images', num_views=5)
    """
        
    row = combined_df.iloc[index]
    
    image_name_1 = row["location_name_1"]
    image_name_2 = row["location_name_2"]
    fig, axes = plt.subplots(2, num_views, figsize=(15, 6))
    fig.suptitle(f"Image Comparison\nOriginal: {image_name_1} | Closest: {image_name_2}", fontsize=14)

    for i in range(num_views):
        file1 = os.path.join(base_path, f"{image_name_1}-{i}.ppm")
        try:
            img1 = Image.open(file1)
            axes[0, i].imshow(img1, cmap="gray")
        except FileNotFoundError:
            axes[0, i].text(0.5, 0.5, "Missing", ha='center', va='center')
        axes[0, i].set_title(f"View {i}")
        axes[0, i].axis('off')

        file2 = os.path.join(base_path, f"{image_name_2}-{i}.ppm")
        try:
            img2 = Image.open(file2)
            axes[1, i].imshow(img2, cmap="gray")
        except FileNotFoundError:
            axes[1, i].text(0.5, 0.5, "Missing", ha='center', va='center')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("From df1", fontsize=12)
    axes[1, 0].set_ylabel("Closest in df2", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()