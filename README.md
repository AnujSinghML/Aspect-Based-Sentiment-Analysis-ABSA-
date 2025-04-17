# Aspect-Based Sentiment Analysis (ABSA) and Predictor Toolkit

Implementation of Aspect-Based Sentiment Analysis (ABSA) using NLP along with a Predictor Toolkit for sentiment prediction.

This work was conducted as part of my internship at **IIIT-Nagpur** under the guidance of **Dr. Pooja Jain**. The research paper explaining the implementation, methodology, and applications of this project is also available.
The explanation and reference can be found in /paper-work/explanantion.md

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Predictor Toolkit](#running-the-predictor-toolkit)
- [Directory Structure](#directory-structure)
- [Dataset Requirements](#dataset-requirements)
- [Results](#results)
- [Contributing](#contributing)

## Installation

To install and set up the project, follow these steps:

### Clone the repository:
```bash
git clone https://github.com/your-username/ABSA-Predictor-Toolkit.git
cd ABSA-Predictor-Toolkit
```

### Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup
Ensure the dataset files are present in the root directory:
- `Restaurants_Train_v2.csv`
- `restaurants-trial.csv`

## Usage

### Training the Model

1. Place the dataset files in the root directory
2. Run the training script:
```bash
python ABSA_NLP.py
```
3. The model will be trained and the best model along with evaluation results will be stored in specified directories
4. **Note:** Training might take a long time depending on your system's performance

### Running the Predictor Toolkit

1. Ensure the best trained model is saved in the `best_model/` directory
2. Run the predictor script:
```bash
python Predictor_Toolkit.py
```
3. Follow the prompts to enter:
   - A sentence for sentiment analysis
   - The aspect you want to analyze
4. The script will output the predicted sentiment for the given aspect and sentence

## Directory Structure

```
ABSA-Predictor-Toolkit/
│── best_model/          # Saved best-performing model
│── checkpoints/         # Model checkpoints during training
│── datasets/            # Contains training and testing datasets
│── results/             # Output and logs from model evaluation
│── ABSA_NLP.py         # Training script
│── Predictor_Toolkit.py # Prediction script
│── requirements.txt     # Required Python dependencies
│── README.md           # Project documentation
```

## Dataset Requirements

### Required Files
- The model is trained using the **SemEval 2014** dataset for aspect-based sentiment analysis
- Required files in root directory before training:
  - `Restaurants_Train_v2.csv` (Training dataset)
  - `restaurants-trial.csv` (Testing dataset)

## Results

- The trained model achieves high accuracy on benchmark datasets
- Results and evaluation metrics are stored in the `results/` directory
- Follow the prompts to input sentences and aspects for sentiment prediction.
The following image shows how the directory should look after training and evaluation.(best moodel and checkpoints are saved)

![image](https://github.com/user-attachments/assets/80086f88-823c-4f6e-8860-c72ebb172e74)

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

Feel free to raise issues for bug fixes or feature requests!


## Contact

- **Author:** [Anuj Sanjay Singh]
- **Email:** [anujsanjaysinghwork@gmail.com]
- **Project Link:** [[https://github.com/your-username/ABSA-Predictor-Toolkit](https://github.com/your-username/ABSA-Predictor-Toolkit](https://github.com/AnujSinghML/Aspect-Based-Sentiment-Analysis-ABSA-.git))

## Acknowledgments

- Dr. Pooja Jain, IIIT-Nagpur

