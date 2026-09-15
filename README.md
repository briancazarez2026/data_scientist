# Machine Learning and Deep Learning Project: Predicting Workplace Burnout

This project applies machine learning and deep learning techniques to analyze and predict **workplace burnout** based on organizational and psychological variables, complemented by practice notebooks on classification models and neural networks.

## Repository Contents

- **`ML.ipynb`** — Exploratory analysis and classic Machine Learning models applied to a burnout dataset.
- **`Deep_learning.ipynb`** — Experiments with SVM kernels on the *Wine* dataset and a hands-on introduction to neural networks with TensorFlow/Keras.

## `ML.ipynb`: Burnout Analysis and Modeling

This notebook loads a custom dataset (uploaded by the user as a `.csv` file) containing variables such as *Work Overload*, *Long Working Hours*, *Role Ambiguity*, *Workplace Harassment*, *Turnover*, *Generalized Anxiety*, and *Nationality*, and follows this workflow:

1. **Data loading and cleaning**: importing the CSV, removing null values, and reviewing descriptive statistics and data types.
2. **Exploratory Data Analysis (EDA)**:
   - Histogram of the Burnout distribution.
   - Box-and-whisker plot of Burnout by nationality, with interquartile range (IQR) calculation.
   - Bar chart showing the number of burnout cases by country (Mexico, India, USA).
   - Pearson correlation matrix across the study's continuous variables.
3. **Multiple linear regression**: predicting Burnout level from variables such as work overload, turnover, harassment, and anxiety, with a visualization of actual vs. predicted values.
4. **Polynomial regression**: fitting degree-2, degree-3, and degree-5 models to the relationship between *Workplace Harassment* and *Burnout*, comparing curve smoothness.
5. **K-Nearest Neighbors (KNN) classification**: converting Burnout into an ordinal categorical variable (Low / Medium / High) using terciles, scaling the features, and training a KNN model.
6. **Support Vector Machine (SVM) classification**: training an SVM model on the same predictors, evaluated with a confusion matrix and classification report.

**Overall goal:** identify which organizational factors best predict burnout and compare the performance of different models (linear regression, polynomial regression, KNN, and SVM) on both regression and classification tasks.

## `Deep_learning.ipynb`: SVM Kernels and Neural Networks

This second notebook has a more educational focus, split into two parts:

### 1. Comparing SVM kernels (Wine dataset)
- Uses scikit-learn's classic `load_wine` dataset (13 chemical features, 3 wine classes).
- Visual explanation of different kernels: **linear, polynomial, RBF, and sigmoid**.
- Trains an SVM with an RBF kernel, evaluated with a confusion matrix and classification report.
- Visualizes the decision boundary in a 2-feature space (alcohol and ash).
- Uses SVR (Support Vector Regression) as an example of applying SVM to a regression problem.

### 2. Introduction to Neural Networks with TensorFlow/Keras
- **Simple regression**: a single-layer neural network that learns to convert Celsius to Fahrenheit, showing the loss curve during training.
- **Network with more hidden layers**: a comparative example illustrating the trade-off between higher accuracy and lower explainability as layers are added.
- **Image classification (Fashion-MNIST)**: a dense neural network that classifies images of clothing items (28x28 pixels, 10 categories), including:
  - Image normalization.
  - Architecture with a `Flatten` layer plus dense layers (ReLU and Softmax).
  - Training with validation, evaluation on the test set, and visualization of accuracy and loss curves per epoch.

## Technologies Used

- **Python** (pandas, numpy)
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn (linear and polynomial regression, KNN, SVM/SVR, evaluation metrics)
- **Deep Learning**: TensorFlow / Keras

## Summary

Overall, the project combines a **real-world case study** (predicting workplace burnout with several ML models) with **hands-on exercises** on SVM kernels and neural network fundamentals, offering a complete walkthrough from exploratory data analysis to deep learning models for image classification.
