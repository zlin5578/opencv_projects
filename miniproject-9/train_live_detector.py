import os
import warnings

# Suppress all warnings (especially from deprecated pkg_resources)
warnings.filterwarnings("ignore")

# Generate absolute path to this script's directory
root = os.path.dirname(os.path.abspath(__file__))

# Construct the path to data.yaml
yaml_path = os.path.join(root, "data.yaml")

# Write the data.yaml file dynamically
with open(yaml_path, "w") as f:
    f.write(f"""train: {os.path.join(root, "dataset/images/train").replace("\\", "/")}
val: {os.path.join(root, "dataset/images/val").replace("\\", "/")}

nc: 2
names: ['Stop Sign', 'Traffic Signal']
""")

# Import YOLOv5 training API
import yolov5.train as train

# Entry point (needed for multiprocessing on Windows)
if __name__ == '__main__':
    train.run(
        img=640,                     # Image size
        batch=16,                    # Batch size
        epochs=50,                   # Number of epochs
        data='data.yaml',            # Path to data config
        weights='yolov5s.pt',        # Base weights
        name='live_detector'         # Experiment name
    )
