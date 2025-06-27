import subprocess
import os

# project target name to ImageNet synset id
classes = {
    # "remote_control": "n03085013",
    # "TV": "n04404412",
    # "Keyboard": "n03085013",  # reuse synset
    # "Cell Phone": "n02992529",
    # "color television": "n03072201",
    # "television": "n06277280",
    # "cable television": "n06278338",
    # "high-definition television": "n06278475",
    # "Remote Control": "n04074963",
    # "remote terminal": "n04075291"
}

# download to root directory
save_root = "./imagenet_dataset"

# downloader.py script path
downloader_script = "downloader.py"

# each class will download 500 images
images_per_class = "500"

# for each class, download images using the downloader script
for class_name, wnid in classes.items():
    print(f"\n=== Downloading class '{class_name}' (wnid: {wnid}) ===")

    # genreate save path for the class
    class_save_path = os.path.join(save_root, class_name)
    os.makedirs(class_save_path, exist_ok=True)

    # construct the command to run the downloader script
    cmd = [
        "python3",
        downloader_script,
        "-data_root", class_save_path,
        "-use_class_list", "True",
        "-class_list", wnid,
        "-images_per_class", images_per_class,
        "-multiprocessing_workers", "4"
    ]

    # execute the command
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)
