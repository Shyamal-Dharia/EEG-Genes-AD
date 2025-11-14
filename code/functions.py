import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

def get_valid_subjects(directory, selected_classes={0, 2}):
    """
    Returns a list of filenames (subject identifiers) whose first label
    is in the selected_classes set.
    """
    files = sorted(os.listdir(directory))
    valid_subjects = []
    for filename in files:
        filepath = os.path.join(directory, filename)
        data = np.load(filepath, allow_pickle=True)
        # Use the first element of the label array as the subject's label.
        label = data["label"][0]
        if label in selected_classes:
            valid_subjects.append(filename)  # using filename as subject id

    return valid_subjects
interested_columns = [
    "BDI", "SES", "RPM", "EHI", "NEO_NEU", "NEO_EXT", "NEO_OPE", "NEO_AGR",
    "NEO_CON", "AUDIT", "MINI-COPE_1", "MINI-COPE_2", "MINI-COPE_3", "MINI-COPE_4", 
    "MINI-COPE_5", "MINI-COPE_6", "MINI-COPE_7", "MINI-COPE_8", "MINI-COPE_9", 
    "MINI-COPE_10", "MINI-COPE_11", "MINI-COPE_12", "MINI-COPE_13", "MINI-COPE_14", 
    "CVLT_1", "CVLT_2", "CVLT_3", "CVLT_4", "CVLT_5", "CVLT_6", "CVLT_7", "CVLT_8", 
    "CVLT_9", "CVLT_10", "CVLT_11", "CVLT_12", "CVLT_13", "dementia_history_parents"

]

# def load_data(directory, target_participant, selected_classes={0, 2}, batch_size=128, shuffle=True, num_of_channels = None):
#     """
#     Loads training and test data based on the target participant.
#     Files with invalid CSV info (i.e. csv_info is None, equals 0, missing required columns, or contains NaN)
#     are skipped.
#     """
#     files = sorted(os.listdir(directory))
#     X_train_list, y_train_list = [], []
#     X_test_list, y_test_list = [], []

#     for filename in files:
#         filepath = os.path.join(directory, filename)
#         data = np.load(filepath, allow_pickle=True)
        
#         # Use the first element of the label array for filtering.
#         label = data["label"][0]
        
#         if label not in selected_classes:
#             continue

#         # Get features. Concatenate HFD and PSD features.
#         X_data = data["HFD_features"]      # shape: (n_windows, n_channels, n_bands)
#         PSD_data = data["PSD_features"]      # shape: (n_windows, n_channels, n_bands)

#         # Check CSV info: if it's None, 0, or missing required columns, skip this file.
#         csv_info = data["csv_info"]
#         if csv_info is None or csv_info == 0:
#             print(f"Skipping file {filename} due to missing csv_info.")
#             continue

#         # Convert csv_info to a dictionary if needed.
#         csv_info = csv_info.item() if hasattr(csv_info, 'item') else csv_info

#         # Try to create an array from the expected columns.
#         try:
#             csv_info_arr = np.array([csv_info[col] for col in interested_columns])
#         except KeyError:
#             print(f"Skipping file {filename} because some columns are missing in csv_info.")
#             continue

#         # Check for NaN values.
#         if np.isnan(csv_info_arr).any():
#             print(f"Skipping file {filename} because csv_info contains NaN values.")
#             continue

#         # If the CSV info is 1D, replicate it for every window.
#         if csv_info_arr.ndim == 1:
#             n_windows = X_data.shape[0]
#             csv_info_arr = np.tile(csv_info_arr, (n_windows, 1))  # now (n_windows, 38)
        
#         # Expand dimensions to add a channel axis, then repeat it to match X_data.
#         csv_info_arr = np.expand_dims(csv_info_arr, axis=1)  # (n_windows, 1, 38)
#         stats_data = np.repeat(csv_info_arr, X_data.shape[1], axis=1)  # (n_windows, n_channels, 38)

#         # Optionally filter by number of channels.
#         if num_of_channels == 16:
#             channels_indices = [0, 31, 29, 1, 2, 8, 7, 23, 24, 25, 18, 12, 13, 15, 16, 17]
#             #pick channels based on the indices
#             X_data = X_data[:, channels_indices, :]
#             PSD_data = PSD_data[:, channels_indices, :]
#             stats_data = stats_data[:, channels_indices, :]

#         elif num_of_channels == 32:
#             channels_indices = [0, 33, 3, 2, 6, 5, 8, 7, 11, 10, 14, 13, 12, 46, 15, 16, 17, 48, 18, 19, 21, 22, 24, 25, 27, 28, 29, 30, 61,31,1,23]
#             X_data = X_data[:, channels_indices, :]
#             PSD_data = PSD_data[:, channels_indices, :]
#             stats_data = stats_data[:, channels_indices, :]

#         # Concatenate features along the third dimension.
#         X_subject = np.concatenate([X_data, PSD_data, stats_data], axis=2)
        
#         # Scale features channel-wise.
#         orig_shape = X_subject.shape
#         X_subject_2d = X_subject.reshape(-1, orig_shape[2])
#         scaler = MinMaxScaler()
#         X_subject_scaled = scaler.fit_transform(X_subject_2d)
#         X_subject = X_subject_scaled.reshape(orig_shape)
        
#         # Get the full label array.
#         y_subject = data["label"]
        
#         # Partition into test and train based on target_participant.
#         if filename == target_participant:
#             X_test_list.append(X_subject)
#             y_test_list.append(y_subject)
#         else:
#             X_train_list.append(X_subject)
#             y_train_list.append(y_subject)

#     if not X_train_list:
#         raise ValueError("No training data found. Check your directory and filtering criteria.")

#     if not X_test_list:
#         raise ValueError(f"No test data found for target participant: {target_participant}. "
#                          "Verify that the participant exists and that the labels match the selected_classes criteria.")
    
#     # Concatenate along the first dimension.
#     X_train = np.concatenate(X_train_list, axis=0)
#     y_train = np.concatenate(y_train_list, axis=0)
#     X_test = np.concatenate(X_test_list, axis=0)
#     y_test = np.concatenate(y_test_list, axis=0)

#     # FIXED: Convert labels based on selected_classes BEFORE counting
#     if selected_classes == {0, 2}:
#         # Map: 0 -> 0, 2 -> 1.
#         y_train = np.where(y_train == 2, 1, y_train)
#         y_test  = np.where(y_test  == 2, 1, y_test)
#     elif selected_classes == {1, 2}:
#         # Map: 1 -> 0, 2 -> 1.
#         # FIXED: Use a temporary copy to avoid overwriting during transformation
#         y_train_new = np.copy(y_train)
#         y_train_new = np.where(y_train == 1, 0, y_train_new)  # 1 -> 0
#         y_train_new = np.where(y_train == 2, 1, y_train_new)  # 2 -> 1
#         y_train = y_train_new
        
#         y_test_new = np.copy(y_test)
#         y_test_new = np.where(y_test == 1, 0, y_test_new)  # 1 -> 0
#         y_test_new = np.where(y_test == 2, 1, y_test_new)  # 2 -> 1
#         y_test = y_test_new
        
#     elif selected_classes == {0, 1, 2}:
#         # Keep original labels
#         pass  # No mapping needed for 3-class

#     # Count classes after label mapping
#     class0 = np.sum(y_train == 0) + np.sum(y_test == 0)
#     class1 = np.sum(y_train == 1) + np.sum(y_test == 1)

#     # Create PyTorch datasets and dataloaders.
#     train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float),
#                                   torch.tensor(y_train, dtype=torch.long))
#     test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float),
#                                   torch.tensor(y_test, dtype=torch.long))
#     if shuffle:
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     else:
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     print(f"Class 0: {class0}, Class 1: {class1}")

#     return train_loader, test_loader

def load_data(directory, target_participant, selected_classes={0, 2}, batch_size=128, shuffle=True, num_of_channels = None):
    """
    Loads training and test data based on the target participant.
    Files with invalid CSV info (i.e. csv_info is None, equals 0, missing required columns, or contains NaN)
    are skipped.
    """
    files = sorted(os.listdir(directory))
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    class0 = 0
    class1 = 0

    for filename in files:
        filepath = os.path.join(directory, filename)
        data = np.load(filepath, allow_pickle=True)
        
        # Use the first element of the label array for filtering.
        label = data["label"][0]
        
        if label not in selected_classes:
            continue

        # Get features. Concatenate HFD and PSD features.
        X_data = data["HFD_features"]      # shape: (n_windows, n_channels, n_bands)
        PSD_data = data["PSD_features"]      # shape: (n_windows, n_channels, n_bands)

        # Check CSV info: if it's None, 0, or missing required columns, skip this file.
        csv_info = data["csv_info"]
        if csv_info is None or csv_info == 0:
            print(f"Skipping file {filename} due to missing csv_info.")
            continue

        # Convert csv_info to a dictionary if needed.
        csv_info = csv_info.item() if hasattr(csv_info, 'item') else csv_info

        # Try to create an array from the expected columns.
        try:
            csv_info_arr = np.array([csv_info[col] for col in interested_columns])
        except KeyError:
            print(f"Skipping file {filename} because some columns are missing in csv_info.")
            continue

        # Check for NaN values.
        if np.isnan(csv_info_arr).any():
            print(f"Skipping file {filename} because csv_info contains NaN values.")
            continue

        # print(csv_info_arr.shape)  # likely prints (38,) indicating a 1D array

        # If the CSV info is 1D, replicate it for every window.
        if csv_info_arr.ndim == 1:
            n_windows = X_data.shape[0]
            csv_info_arr = np.tile(csv_info_arr, (n_windows, 1))  # now (n_windows, 38)
        
        # Expand dimensions to add a channel axis, then repeat it to match X_data.
        csv_info_arr = np.expand_dims(csv_info_arr, axis=1)  # (n_windows, 1, 38)
        stats_data = np.repeat(csv_info_arr, X_data.shape[1], axis=1)  # (n_windows, n_channels, 38)

        # Optionally filter by number of channels.
        if num_of_channels == 16:
            channels_indices = [0, 31, 29, 1, 2, 8, 7, 23, 24, 25, 18, 12, 13, 15, 16, 17]
            #pick channels based on the indices
            X_data = X_data[:, channels_indices, :]
            PSD_data = PSD_data[:, channels_indices, :]
            stats_data = stats_data[:, channels_indices, :]

        elif num_of_channels == 32:
            channels_indices = [0, 33, 3, 2, 6, 5, 8, 7, 11, 10, 14, 13, 12, 46, 15, 16, 17, 48, 18, 19, 21, 22, 24, 25, 27, 28, 29, 30, 61,31,1,23]
            X_data = X_data[:, channels_indices, :]
            PSD_data = PSD_data[:, channels_indices, :]
            stats_data = stats_data[:, channels_indices, :]

        # Concatenate features along the third dimension.
        X_subject = np.concatenate([X_data, PSD_data, stats_data], axis=2)
        
        # Scale features channel-wise.
        orig_shape = X_subject.shape
        X_subject_2d = X_subject.reshape(-1, orig_shape[2])
        scaler = MinMaxScaler()
        X_subject_scaled = scaler.fit_transform(X_subject_2d)
        X_subject = X_subject_scaled.reshape(orig_shape)
        
        # Get the full label array.
        y_subject = data["label"]
        
        # Partition into test and train based on target_participant.
        if filename == target_participant:
            X_test_list.append(X_subject)
            y_test_list.append(y_subject)
        else:
            X_train_list.append(X_subject)
            y_train_list.append(y_subject)
        

        # print(y_subject[0])
        if y_subject[0] == 0:
            class0 += 1
        else:
            class1 += 1

    if not X_train_list:
        raise ValueError("No training data found. Check your directory and filtering criteria.")

    if not X_test_list:
        raise ValueError(f"No test data found for target participant: {target_participant}. "
                         "Verify that the participant exists and that the labels match the selected_classes criteria.")
    
    # Concatenate along the first dimension.
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Optionally convert labels based on selected_classes.
    if selected_classes == {0, 2}:
        # Map: 0 -> 0, 2 -> 1.
        y_train = np.where(y_train == 2, 1, y_train)
        y_test  = np.where(y_test  == 2, 1, y_test)
    elif selected_classes == {1, 2}:
        # Map: 1 -> 0, 2 -> 1.
        y_train = np.where(y_train == 1, 0, y_train)
        y_train = np.where(y_train == 2, 1, y_train)
        y_test  = np.where(y_test == 1, 0, y_test)
        y_test  = np.where(y_test == 2, 1, y_test)

    elif selected_classes == {0, 1, 2}:
        # Example mapping if desired.
        y_train = np.where(y_train == 1, 1, y_train)
        y_test  = np.where(y_test == 1, 1, y_test)
        y_train = np.where(y_train == 2, 2, y_train)
        y_test  = np.where(y_test == 2, 2, y_test)

    # Create PyTorch datasets and dataloaders.
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float),
                                  torch.tensor(y_test, dtype=torch.long))
    if shuffle:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print(f"Class 0: {class0}, Class 1: {class1}")
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))


    return train_loader, test_loader

def load_data_ablation(directory, target_participant, selected_classes={0, 2}, batch_size=128, shuffle=True, num_of_channels=None, feature_types=['hfd', 'psd', 'csv']):
    """
    Loads training and test data based on the target participant with ablation study support.
    
    Args:
        directory: Directory containing data files
        target_participant: Target participant for testing
        selected_classes: Classes to include in the dataset
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle training data
        num_of_channels: Number of channels to use (16, 32, or None for all)
        feature_types: List of feature types to include. Options:
                      - 'hfd': HFD features
                      - 'psd': PSD features  
                      - 'csv': Psychometric/CSV features
                      Example: ['hfd', 'psd'] for HFD+PSD only
    
    Returns:
        train_loader, test_loader
    """
    # Validate feature_types input
    valid_features = {'hfd', 'psd', 'csv'}
    if not isinstance(feature_types, list):
        raise ValueError("feature_types must be a list")
    if not all(ft in valid_features for ft in feature_types):
        raise ValueError(f"feature_types must contain only: {valid_features}")
    if len(feature_types) == 0:
        raise ValueError("feature_types cannot be empty")
    
    files = sorted(os.listdir(directory))
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print(f"Ablation study with features: {feature_types}")

    for filename in files:
        filepath = os.path.join(directory, filename)
        data = np.load(filepath, allow_pickle=True)
        
        # Use the first element of the label array for filtering.
        label = data["label"][0]
        
        if label not in selected_classes:
            continue

        # Get features
        X_data = data["HFD_features"]      # shape: (n_windows, n_channels, n_bands)
        PSD_data = data["PSD_features"]    # shape: (n_windows, n_channels, n_bands)

        # Handle CSV features
        csv_info = data["csv_info"]
        skip_file = False
        
        if 'csv' in feature_types:
            # Check CSV info: if it's None, 0, or missing required columns, skip this file.
            if csv_info is None or csv_info == 0:
                print(f"Skipping file {filename} due to missing csv_info (required for CSV features).")
                skip_file = True
                continue

            # Convert csv_info to a dictionary if needed.
            csv_info = csv_info.item() if hasattr(csv_info, 'item') else csv_info

            # Try to create an array from the expected columns.
            try:
                csv_info_arr = np.array([csv_info[col] for col in interested_columns])
            except KeyError:
                print(f"Skipping file {filename} because some columns are missing in csv_info (required for CSV features).")
                skip_file = True
                continue

            # Check for NaN values.
            if np.isnan(csv_info_arr).any():
                print(f"Skipping file {filename} because csv_info contains NaN values (required for CSV features).")
                skip_file = True
                continue

            # If the CSV info is 1D, replicate it for every window.
            if csv_info_arr.ndim == 1:
                n_windows = X_data.shape[0]
                csv_info_arr = np.tile(csv_info_arr, (n_windows, 1))  # now (n_windows, 38)
            
            # Expand dimensions to add a channel axis, then repeat it to match X_data.
            csv_info_arr = np.expand_dims(csv_info_arr, axis=1)  # (n_windows, 1, 38)
            stats_data = np.repeat(csv_info_arr, X_data.shape[1], axis=1)  # (n_windows, n_channels, 38)

        if skip_file:
            continue

        # Optionally filter by number of channels.
        if num_of_channels == 16:
            channels_indices = [0, 31, 29, 1, 2, 8, 7, 23, 24, 25, 18, 12, 13, 15, 16, 17]
            X_data = X_data[:, channels_indices, :]
            PSD_data = PSD_data[:, channels_indices, :]
            if 'csv' in feature_types:
                stats_data = stats_data[:, channels_indices, :]

        elif num_of_channels == 32:
            channels_indices = [0, 33, 3, 2, 6, 5, 8, 7, 11, 10, 14, 13, 12, 46, 15, 16, 17, 48, 18, 19, 21, 22, 24, 25, 27, 28, 29, 30, 61, 31, 1, 23]
            X_data = X_data[:, channels_indices, :]
            PSD_data = PSD_data[:, channels_indices, :]
            if 'csv' in feature_types:
                stats_data = stats_data[:, channels_indices, :]

        # Concatenate selected features along the third dimension
        feature_list = []
        
        if 'hfd' in feature_types:
            feature_list.append(X_data)
            
        if 'psd' in feature_types:
            feature_list.append(PSD_data)
            
        if 'csv' in feature_types:
            feature_list.append(stats_data)
        
        # Concatenate selected features
        X_subject = np.concatenate(feature_list, axis=2)
        
        # Scale features channel-wise.
        orig_shape = X_subject.shape
        X_subject_2d = X_subject.reshape(-1, orig_shape[2])
        scaler = MinMaxScaler()
        X_subject_scaled = scaler.fit_transform(X_subject_2d)
        X_subject = X_subject_scaled.reshape(orig_shape)
        
        # Get the full label array.
        y_subject = data["label"]
        
        # Partition into test and train based on target_participant.
        if filename == target_participant:
            X_test_list.append(X_subject)
            y_test_list.append(y_subject)
        else:
            X_train_list.append(X_subject)
            y_train_list.append(y_subject)

    if not X_train_list:
        raise ValueError("No training data found. Check your directory and filtering criteria.")

    if not X_test_list:
        raise ValueError(f"No test data found for target participant: {target_participant}. "
                         "Verify that the participant exists and that the labels match the selected_classes criteria.")
    
    # Concatenate along the first dimension.
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Convert labels based on selected_classes.
    if selected_classes == {0, 2}:
        # Map: 0 -> 0, 2 -> 1.
        y_train = np.where(y_train == 2, 1, y_train)
        y_test  = np.where(y_test  == 2, 1, y_test)
    elif selected_classes == {1, 2}:
        # Map: 1 -> 0, 2 -> 1.
        y_train_new = np.copy(y_train)
        y_train_new = np.where(y_train == 1, 0, y_train_new)  # 1 -> 0
        y_train_new = np.where(y_train == 2, 1, y_train_new)  # 2 -> 1
        y_train = y_train_new
        
        y_test_new = np.copy(y_test)
        y_test_new = np.where(y_test == 1, 0, y_test_new)  # 1 -> 0
        y_test_new = np.where(y_test == 2, 1, y_test_new)  # 2 -> 1
        y_test = y_test_new
        
    elif selected_classes == {0, 1, 2}:
        # Keep original labels for 3-class
        pass

    # Create PyTorch datasets and dataloaders.
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float),
                                  torch.tensor(y_test, dtype=torch.long))
    
    if shuffle:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print feature information
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Feature composition: {feature_types}")
    print("Label distribution:")
    print(f"  Train: {np.unique(y_train, return_counts=True)}")
    print(f"  Test:  {np.unique(y_test, return_counts=True)}")

    return train_loader, test_loader










def load_data_SD(directory, test_split, seed, selected_classes={0, 2}, batch_size=128, shuffle=True, num_of_channels = None):
    """
    Loads training and test data based on the target participant.
    Files with invalid CSV info (i.e. csv_info is None, equals 0, missing required columns, or contains NaN)
    are skipped.
    """
    files = sorted(os.listdir(directory))
    X_train_list, y_train_list = [], []

    for filename in files:
        filepath = os.path.join(directory, filename)
        data = np.load(filepath, allow_pickle=True)
        
        # Use the first element of the label array for filtering.
        label = data["label"][0]
        if label not in selected_classes:
            continue

        # Get features. Concatenate HFD and PSD features.
        X_data = data["HFD_features"]      # shape: (n_windows, n_channels, n_bands)
        PSD_data = data["PSD_features"]      # shape: (n_windows, n_channels, n_bands)

        # Check CSV info: if it's None, 0, or missing required columns, skip this file.
        csv_info = data["csv_info"]
        if csv_info is None or csv_info == 0:
            print(f"Skipping file {filename} due to missing csv_info.")
            continue

        # Convert csv_info to a dictionary if needed.
        csv_info = csv_info.item() if hasattr(csv_info, 'item') else csv_info

        # Try to create an array from the expected columns.
        try:
            csv_info_arr = np.array([csv_info[col] for col in interested_columns])
        except KeyError:
            print(f"Skipping file {filename} because some columns are missing in csv_info.")
            continue

        # Check for NaN values.
        if np.isnan(csv_info_arr).any():
            print(f"Skipping file {filename} because csv_info contains NaN values.")
            continue

        # print(csv_info_arr.shape)  # likely prints (38,) indicating a 1D array

        # If the CSV info is 1D, replicate it for every window.
        if csv_info_arr.ndim == 1:
            n_windows = X_data.shape[0]
            csv_info_arr = np.tile(csv_info_arr, (n_windows, 1))  # now (n_windows, 38)
        
        # Expand dimensions to add a channel axis, then repeat it to match X_data.
        csv_info_arr = np.expand_dims(csv_info_arr, axis=1)  # (n_windows, 1, 38)
        stats_data = np.repeat(csv_info_arr, X_data.shape[1], axis=1)  # (n_windows, n_channels, 38)

        # Optionally filter by number of channels.
        if num_of_channels == 16:
            channels_indices = [0, 31, 29, 1, 2, 8, 7, 23, 24, 25, 18, 12, 13, 15, 16, 17]
            #pick channels based on the indices
            X_data = X_data[:, channels_indices, :]
            PSD_data = PSD_data[:, channels_indices, :]
            stats_data = stats_data[:, channels_indices, :]

        elif num_of_channels == 32:
            channels_indices = [0, 33, 3, 2, 6, 5, 8, 7, 11, 10, 14, 13, 12, 46, 15, 16, 17, 48, 18, 19, 21, 22, 24, 25, 27, 28, 29, 30, 61,31,1,23]
            X_data = X_data[:, channels_indices, :]
            PSD_data = PSD_data[:, channels_indices, :]
            stats_data = stats_data[:, channels_indices, :]

        # Concatenate features along the third dimension.
        X_subject = np.concatenate([X_data, PSD_data, stats_data], axis=2)
        
        # Scale features channel-wise.
        orig_shape = X_subject.shape
        X_subject_2d = X_subject.reshape(-1, orig_shape[2])
        scaler = MinMaxScaler()
        X_subject_scaled = scaler.fit_transform(X_subject_2d)
        X_subject = X_subject_scaled.reshape(orig_shape)
        
        # Get the full label array.
        y_subject = data["label"]
        
        # Partition into test and train based on target_participant.
    
        X_train_list.append(X_subject)
        y_train_list.append(y_subject)

    if not X_train_list:
        raise ValueError("No training data found. Check your directory and filtering criteria.")

    
    # Concatenate along the first dimension.
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_split, random_state=seed)

    # Optionally convert labels based on selected_classes.
    if selected_classes == {0, 2}:
        # Map: 0 -> 0, 2 -> 1.
        y_train = np.where(y_train == 2, 1, y_train)
        y_test  = np.where(y_test  == 2, 1, y_test)
    elif selected_classes == {1, 2}:
        # Map: 1 -> 0, 2 -> 1.
        y_train = np.where(y_train == 1, 0, y_train)
        y_train = np.where(y_train == 2, 1, y_train)
        y_test  = np.where(y_test == 1, 0, y_test)
        y_test  = np.where(y_test == 2, 1, y_test)
    elif selected_classes == {0, 1, 2}:
        # Example mapping if desired.
        y_train = np.where(y_train == 1, 1, y_train)
        y_test  = np.where(y_test == 1, 1, y_test)
        y_train = np.where(y_train == 2, 2, y_train)
        y_test  = np.where(y_test == 2, 2, y_test)

    # Create PyTorch datasets and dataloaders.
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float),
                                  torch.tensor(y_test, dtype=torch.long))
    if shuffle:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader













def cross_entropy_soft(pred):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    loss = torch.mean(torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss

def attentive_entropy(pred, pred_domain):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)

    # attention weight
    entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
    weights = 1 + entropy

    # attentive entropy
    loss = torch.mean(weights * torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss

import torch.nn.functional as F
def dis_MCD(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2, dim=1)))



def train(model, dataloader,  loss_fn, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in dataloader:
        
        X, y = X.to(device), y.to(device)
        #get the target loader data first batch
        optimizer.zero_grad()
        outputs, wei = model(X)
        loss = loss_fn(outputs, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() #* X.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def eval(model, dataloader, loss_fn, device):
    """
    Evaluate the model on the test data.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs, wei = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


