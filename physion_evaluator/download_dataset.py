# download_dataset.py
import os
import subprocess

import os
import subprocess

def download_and_extract_dataset(dataset_dir):
    dataset_url = "https://storage.googleapis.com/physion-dataset/physion_dataset.zip"
    dataset_zip = os.path.join(dataset_dir, "physion_dataset.zip")

    # Create dataset directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Download the dataset zip file
    subprocess.run(["wget", dataset_url, "-O", dataset_zip], check=True)

    # Unzip the dataset into the dataset directory
    print(f"Unzipping {dataset_zip} into {dataset_dir}...")
    subprocess.run(["unzip", "-q", dataset_zip, "-d", dataset_dir], check=True)

    # Remove the zip file
    print(f"Deleting {dataset_zip}...")
    os.remove(dataset_zip)
    print("Dataset setup completed.")
