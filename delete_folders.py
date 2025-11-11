import os
import shutil

path = r"/mnt/cs/cs153/datasets/HaGRIDv2_dataset_512"
deleted_path = r"/mnt/home/nvithiananthan/courses/cs153/cv-music-gestures/deleted_folders.txt"

with open(deleted_path, 'r') as f:
    deleted_folders = [line.strip() for line in f if line.strip()]


for folder in deleted_folders:
    folder_path = os.path.join(path, folder)

    if os.path.isdir(folder_path):
        print(f"deleting {folder_path}")
        shutil.rmtree(folder_path)
