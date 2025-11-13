import os
import shutil

og_path = "/mnt/cs/cs153/datasets/HaGRIDv2_dataset_512"
main_folder = "/mnt/cs/cs153/datasets/music_prod_gestures"

mapping = {"one":["one", "mute"], 
           "two": ["peace", "two_up"], 
           "three":["three"], 
           "four":["four"], 
           "five":["palm", "stop"], 
           "reset": ["fist"],
           "synth":["three_gun"],
           "thumbs_up":["like"],
            "rock":["rock"]}

dest_folders = mapping.keys()

for dest_folder in dest_folders:
    source_folders = mapping[dest_folder]

    dest_path = os.path.join(main_folder, dest_folder)
    os.makedirs(dest_path, exist_ok=True)
    for source in source_folders:
        print(f"copying {source} to {dest_folder}")
        source_path = os.path.join(og_path, source)
        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
