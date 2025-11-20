import splitfolders

# Define input and output paths
input_folder = r"/mnt/cs/cs153/datasets/music_prod_gestures"
output_folder = r"/mnt/cs/cs153/datasets/music_gestures"

splitfolders.ratio(input_folder, output=output_folder, seed=153, ratio=(0.7, 0.15, 0.15))