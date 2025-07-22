# Music Genre Classification using CNN, MFCC, and Mel Spectrograms (GTZAN Dataset)

This project focuses on **classifying music tracks into 10 genres** using a **deep Convolutional Neural Network (CNN)**. The dataset is preprocessed using **Mel Spectrograms** and **Mel-Frequency Cepstral Coefficients (MFCC)** to create robust feature representations of audio signals.  

---

## **Project Overview**
- **Goal:** Automatically classify audio tracks into one of 10 music genres (Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock).
- **Dataset:** [GTZAN Dataset](http://marsyas.info/downloads/datasets.html) – 1,000 audio clips (30 seconds each).
- **Approach:**  
  - Audio preprocessing using **Librosa** to extract MFCCs and Mel Spectrograms.
  - Conversion of audio segments into spectrogram images.
  - Training a **deep CNN** with **stacked Conv2D–ReLU–MaxPooling blocks**, dropout, and batch normalization.

---

## **Key Features**
- **Audio Preprocessing:**  
  - Audio chunked into 4-second segments with 2-second overlap.
  - MFCC and Mel Spectrogram features extracted using Librosa.
- **Model Architecture:**  
  - Multiple convolutional and pooling layers with ReLU activations.
  - Dense layers and softmax output for multi-class classification.
  - Regularization using **dropout** and **early stopping**.
- **Performance:**  
  - **Training Accuracy:** 99.2%  
  - **Validation Accuracy:** 90.5%  
  - **Test Accuracy:** 89.9%  
  - **Macro F1-score:** 0.91  

---

## **Classification Report**
| Genre       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Blues       | 0.76      | 0.97   | 0.85     | 302     |
| Classical   | 0.94      | 0.96   | 0.95     | 298     |
| Country     | 0.86      | 0.87   | 0.87     | 317     |
| Disco       | 0.95      | 0.88   | 0.92     | 312     |
| HipHop      | 0.95      | 0.94   | 0.95     | 277     |
| Jazz        | 0.93      | 0.95   | 0.94     | 311     |
| Metal       | 0.97      | 0.97   | 0.97     | 302     |
| Pop         | 0.92      | 0.89   | 0.90     | 289     |
| Reggae      | 0.94      | 0.88   | 0.91     | 296     |
| Rock        | 0.89      | 0.74   | 0.81     | 291     |
**Overall Accuracy:** 0.91  

---

## **Tech Stack**
- **Languages:** Python  
- **Libraries:** TensorFlow/Keras, Librosa, NumPy, Pandas, Matplotlib  
- **Tools:** Jupyter Notebook, Git, Google Colab

---

## **Repository Structure**
