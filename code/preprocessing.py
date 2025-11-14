import pandas as pd
import numpy as np
import glob
import re
import mne
from mne.preprocessing import ICA
import os

def get_start_end(events, start_code=4, end_code=11):
    event_codes = events[:, 2]
    start_idx = end_idx = None

    if start_code in event_codes:
        start_idx = events[np.where(event_codes == start_code)[0][0], 0]
    
    if end_code in event_codes:
        end_idx = events[np.where(event_codes == end_code)[0][0], 0]
    
    return start_idx, end_idx

file_paths = glob.glob("/home/s.dharia-ra/Shyamal/PEARL/sub-*/eeg/*_task-rest_eeg.vhdr")
print(f"Number of EEG files found: {len(file_paths)}")

def extract_participant_id(path):
    match = re.search(r'/sub-\d+', path)
    if match:
        return match.group().split('/')[-1]
    else:
        return None

participant_ids_with_files = set(filter(None, [extract_participant_id(path) for path in file_paths]))
print(f"Participant IDs with EEG files: {participant_ids_with_files}")
print(f"Total unique participants with EEG files: {len(participant_ids_with_files)}")

participants = pd.read_csv("./participants.tsv", sep="\t")
participants_with_files = participants[participants['participant_id'].isin(participant_ids_with_files)].copy()
print(f"Participants with corresponding EEG files: {len(participants_with_files)}")

missing_ids = participant_ids_with_files - set(participants['participant_id'])
if missing_ids:
    print(f"Warning: The following participant IDs have EEG files but are missing in participants.tsv: {missing_ids}")
else:
    print("All participant IDs with EEG files are present in participants.tsv.")

# Exclude APOE-ε2 carriers
participants_filtered = participants_with_files[~participants_with_files['APOE_haplotype'].str.lower().str.contains('e2')].copy()
print(f"Participants after excluding APOE-ε2 carriers: {len(participants_filtered)}")

# Map participant IDs to group labels
group_mapping = {'N': 0, 'A+P-': 1, 'A+P+': 2}
classified_participants = pd.read_csv("./filtered_participants.tsv", sep="\t")
participant_id_to_group = dict(zip(classified_participants['participant_id'], classified_participants['Group'].map(group_mapping)))

print("Starting EEG preprocessing...")

output_dir = "resting_state_preprocessed_data"
os.makedirs(output_dir, exist_ok=True)

b_freq = 1.0  # Bandpass filter lower frequency
u_freq = 45.0 # Bandpass filter upper frequency

adjust_samples = 361291  # Adjust this value based on your data's sampling rate and desired time window

for idx, file_path in enumerate(file_paths):
    participant_id = extract_participant_id(file_path)
    
    if participant_id not in participant_id_to_group:
        print(f"Participant {participant_id} not in classified groups. Skipping.")
        continue
    
    group_label = participant_id_to_group[participant_id]
    
    # Define the output file path
    output_file = os.path.join(output_dir, f"{participant_id}_preprocessed.npz")
    
    # Check if the file has already been processed
    if os.path.exists(output_file):
        print(f"\nProcessing file {idx+1}/{len(file_paths)}: {file_path} | Group: {group_label}")
        print(f"Preprocessed data for {participant_id} already exists. Skipping.")
        continue
    
    print(f"\nProcessing file {idx+1}/{len(file_paths)}: {file_path} | Group: {group_label}")
    
    try:
        # Load raw EEG data
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
        events, events_id = mne.events_from_annotations(raw)
        
        # Get start and end indices
        start_sample, end_sample = get_start_end(events, start_code=4, end_code=11)
        
        if start_sample is not None and end_sample is not None:
            # Both start and end events found
            raw = raw.crop(tmin=start_sample / raw.info["sfreq"], tmax=end_sample / raw.info["sfreq"])
        elif start_sample is not None:
            # Only start event found
            new_end_sample = start_sample + adjust_samples
            if new_end_sample > raw.n_times:
                new_end_sample = raw.n_times
            raw = raw.crop(tmin=start_sample / raw.info["sfreq"], tmax=new_end_sample / raw.info["sfreq"])
            print(f"Only start event found. Cropped from start index {start_sample} to {new_end_sample}.")
        elif end_sample is not None:
            # Only end event found
            new_start_sample = end_sample - adjust_samples
            if new_start_sample < 0:
                new_start_sample = 0
            raw = raw.crop(tmin=new_start_sample / raw.info["sfreq"], tmax=end_sample / raw.info["sfreq"])
            print(f"Only end event found. Cropped from start index {new_start_sample} to end index {end_sample}.")
        else:
            # Neither start nor end event found
            print("No valid start or end events found. Skipping this subject.")
            continue
    
        # Apply bandpass filter
        raw = raw.filter(l_freq=b_freq, h_freq=None)
        
        # Apply notch filter at 50 Hz (common power line frequency)
        raw.notch_filter(freqs=50)
        
        # Resample the data to 256 Hz
        raw = raw.resample(sfreq=256)
        
        # raw.set_channel_types({'FCz': 'eeg'})
        # Perform ICA for artifact removal
        ica = ICA(n_components=42, method="picard", max_iter="auto", random_state=97)
        ica.fit(raw)
        
        # Automatically detect muscle artifacts
        muscle_idx_auto, muscle_scores = ica.find_bads_muscle(raw)
        if len(muscle_idx_auto) > 0:
            ica.plot_components(picks=muscle_idx_auto, title="Muscle Artifact Components", colorbar=True)
        print(f"Automatically found muscle artifact ICA components: {muscle_idx_auto}")
        
        # Automatically detect ocular (EOG) artifacts
        eye_idx, eog_scores = ica.find_bads_eog(raw, ch_name=["Fp2", "Fp1"], threshold=0.5, measure="correlation")
        if len(eye_idx) > 0:
            ica.plot_components(picks=eye_idx, title="Ocular Artifact Components", colorbar=True)
        print(f"Automatically found ocular (EOG) artifact ICA components: {eye_idx}")
        
        # Combine components to exclude
        exclude_list = list(set(muscle_idx_auto + eye_idx))
        print(f"Excluding ICA components: {exclude_list}")
        
        # Apply ICA exclusion
        ica.apply(raw, exclude=exclude_list)
        
        # Apply high-pass filter after ICA
        raw = raw.filter(l_freq=None, h_freq=u_freq)
        raw, ref_data = mne.set_eeg_reference(raw, ref_channels='average', projection=False)
        fcz_data = ref_data[np.newaxis, :]  # shape becomes (1, n_times)
        
        

        # Extract the data as a NumPy array
        data = raw.get_data()
        
        data = np.vstack((data, fcz_data))
        print("###############################")
        print(data.shape)

        # Save the preprocessed data and label
        np.savez_compressed(
            output_file,
            data=data,
            label=group_label
        )
        print(f"Preprocessed data saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred while processing {participant_id}: {e}")
        continue

print("ICA-based artifact removal pipeline complete.")
