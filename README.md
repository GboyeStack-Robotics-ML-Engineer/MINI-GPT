# MINI-GPT Project

## Overview

This project is a mini implementation of a GPT (Generative Pre-trained Transformer) model using PyTorch and Ray for distributed training. The codebase includes data preprocessing, model training, and evaluation.

## Files

- `TRAIN.py`: Main script for training the GPT model.
- `DATA/Train.csv`: Training data in CSV format.
- `DATA/Test.csv`: Test data in CSV format.
- `Test.ipynb`: Jupyter notebook for testing and running the training script.

## Dependencies

- Python 3.7+
- PyTorch
- Ray
- Transformers (Hugging Face)
- Accelerate
- Pandas
- Evaluate

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/GboyeStack-Robotics-ML-Engineer/MINI-GPT.git
   cd MINI-GPT
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, run the `TRAIN.py` script with the appropriate arguments:

```sh
python TRAIN.py --use_gpu=True --trainer_resources CPU=2 GPU=0 --num_workers=2 --resources_per_worker CPU=1 GPU=1
```

### Jupyter Notebook

You can also use the `Test.ipynb` notebook to run the training script and test the model.

## Code Explanation

### `TRAIN.py`

- **Imports**: The script imports necessary libraries including PyTorch, Ray, and Hugging Face Transformers.
- **TrainGpt Function**: This function handles the training loop, including data loading, model initialization, and training steps.
- **Main Block**: Parses command-line arguments and initializes the Ray trainer for distributed training.

### `Test.ipynb`

- **Setup**: Clones the repository and installs dependencies.
- **Training**: Runs the training script with specified parameters.
- **Testing**: Contains code for testing the trained model.

## Data

The training and test data are stored in CSV files located in the `DATA` directory. The data is loaded and processed using the `DataGenerator` class.

## Model

The model is defined in the `GPT.py` file and uses the T5 architecture from Hugging Face Transformers.

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License.
