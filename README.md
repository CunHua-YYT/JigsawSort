

# requirements

einops==0.6.0
numpy==1.23.5
pandas==1.3.1
path==16.6.0
path.py==12.5.0
pathtools==0.1.2
Pillow==9.4.0
PyYAML
timm==0.3.2
tqdm==4.64.1

And the torch framework.

File Structure
The configs directory contains YAML files, which can be modified to change parameters.

The output_dir directory contains trained weight files in .pth format and backup configuration files.

The utils.py file contains commonly used utility functions.

The engine.py file contains training-related functions.

The datasets.py file handles the division of training and testing datasets.

The train.py file contains the training code.

The main.py file loads model weights and performs predictions.

The argparse parameters store only some experiment-independent parameters, including the configuration file path, and the main purpose is to store result-related paths. After entering the main function, they are only used at the beginning. Then, the parameters from the configuration file (yaml) are reloaded using the set_defaults() function.

The args parameters persist throughout the process and are also backed up with the results.

Usage
Run infer.py directly for inference and obtain the time taken for prediction on 64-block and 100-block.

Specify the file path and image name to be predicted directly below. The file_path is the directory path, and single_img_name is the name of the image to be predicted. You can define it in the 1.yaml file in the configs directory or modify it yourself.