# Facial Expression Recognition using CNN & GAN
Facial Expression Recognition (FER) is a critical component in human-computer interaction systems, used in applications ranging from mental health monitoring to smart surveillance. This project implements a deep learning-based FER system using **Convolutional Neural Networks (CNNs)** for classification and **Generative Adversarial Networks (GANs)** for data augmentation to improve model robustness and performance.

## Features

- Emotion classification into key categories (e.g., Angry, Happy, Sad, Surprise, etc.)
- CNN-based architecture for expression recognition
- GAN-based image augmentation to handle class imbalance
- Preprocessing pipeline with face detection and alignment
- Visualization tools for predictions and augmented samples
  
---
## Dataset

We used the **[FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)** from Kaggle

Each image is preprocessed using:
- Grayscale conversion
- Face alignment 

---
## Model Architecture

### CNN Model:
- Conv2D → BatchNorm → ReLU → MaxPooling
- Dropout + Dense layers
- Softmax activation for classification

### GAN Model:
- Generator: Fully connected → Conv2DTranspose layers
- Discriminator: Conv2D → LeakyReLU → Dense
- Trained with adversarial loss

---
## How to Run the Project
### Prerequisites
- Python 3.7 or above
- pip (Python package manager)
- GPU (optional but recommended for training GANs)

1. Clone the Repository
Download the project to your local machine by cloning this GitHub repository.

2. Download and Organize the Dataset
Get the FER-2013 dataset from Kaggle and organize it into train and test folders by emotion categories inside an images directory.

3. Set Up the Environment
Create a virtual environment (optional) and install all required dependencies using the requirements.txt file.

4. Train the CNN Classifier
Run the CNN model script to train the emotion classification model on the dataset. Adjust image size or paths if needed.

5. Generate Images using GAN
Run the GAN model script to generate synthetic facial expression images. Generated outputs will be saved locally.

6. Visualize and Test
Use the visualization script to preview emotion-specific samples, or test the model in real-time if webcam detection is implemented.

---
## Sample Results


---
## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- OpenCV (optional for real-time testing)

---
## Contributors
- J.Surya Teja
- K.Shriya Bushan
- Pavan
  
