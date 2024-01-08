# Create Your Own Image Classifier

## AI Plant Classifier Project

## Introduction
Welcome to the AI Plant Classifier Project! This is an Artificial Intelligence (AI) application capable of detecting over 100 different categories of plants. The project utilizes deep learning methodologies to train a neural network that can accurately classify plant species from images.

## Getting Started

### Setting Up the Environment with Anaconda

1. **Install Anaconda**: Download and install Anaconda from the [official site](https://www.anaconda.com/products/individual). Anaconda simplifies package management and deployment for Python.

2. **Create a Conda Environment**: After installing Anaconda, create a new environment for the project.

    ```bash
    conda env create -f environment.yml
    ```

    Replace `plantenv` with the name you wish to give your environment.

3. **Activate the Environment**:

    ```bash
    conda activate udacity
    ```

### Project Structure

The project includes key Python scripts:

- `train.py` - For training the model on the dataset.
- `predict.py` - For making predictions using the trained model.

### Train
- Basic usage: python train.py data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
  - Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory```
  - Choose architecture: ```python train.py data_dir --arch "vgg13" ```
  - Set hyperparameters: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20```
  -  Use GPU for training: ```python train.py data_dir --gpu```

### predict
- Basic usage: ```python predict.py /path/to/image checkpoint```
- Options:
  - Return top K most likely classes: ``python predict.py input checkpoint --top_k 3``
  - Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
  - Use GPU for inference: ```python predict.py input checkpoint --gpu```

### Training the Classifier

1. **Prepare Your Dataset**: Organize your plant images in a directory structure suitable for training. Typically, this means structuring directories by label or category.


