# MNIST-Classification-using-ML
# README - MNIST Handwritten Digit Classification Project

## Project Overview
This project is focused on **handwritten digit classification** using the **MNIST dataset** and a **Random Forest Classifier**. The objective is to train a model that can accurately recognize digits (0-9) without using deep learning techniques, achieving an **F1-score greater than 0.95**.

The submission includes:
- **A Professional Report (PDF)** summarizing the dataset, methodology, results, and conclusions.
- **Python Code Files (`.ipynb` and `.py`)** for model training, testing, and evaluation.
- **A README File** explaining how to run the project and interpret results.

---

## Contents of the ZIP File
The submitted ZIP file contains the following files:

| File Name                     | Description |
|--------------------------------|-------------|
| `mnist_classification_report.pdf` | Detailed report covering project methodology and findings. |
| `mnist_classification.ipynb`   | Jupyter Notebook with code, explanations, and outputs. |
| `mnist_classification.py`      | Python script for training and evaluating the model. |
| `README.md`                    | Instructions on how to run the project. |

---

## Running the Code
The project can be executed in **Google Colab** or on a **local machine** with Python installed.

### Option 1: Run in Google Colab
1. Upload the dataset ZIP file to Google Colab.
2. Open and run `mnist_classification.ipynb` step by step.
3. View outputs, including accuracy, confusion matrix, and sample predictions.

### Option 2: Run Locally (Python Required)
1. Extract the dataset.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Run the Python script:
   ```bash
   python mnist_classification.py
   ```

---

## Expected Outputs
- **Accuracy Scores:** Performance metrics on validation and test sets.
- **Classification Reports:** Precision, recall, and F1-score for each digit.
- **Confusion Matrix:** Visual representation of misclassifications.
- **Sample Predictions:** Display of test images with predicted labels.

---

## How to Read the Report
The **PDF report** contains:
- **Introduction** – Overview of the project objectives.
- **Dataset and Model Description** – Explanation of MNIST data and selected algorithms.
- **Solutions, Findings, and Results** – Performance evaluation with tables and graphs.
- **Discussion and Conclusion** – Interpretation of results and future improvements.
- **## References** – Sources used in the project.

---

## Additional Notes
- The dataset used is **MNIST**, available on Kaggle.
- The classifier used is **Random Forest with 100 estimators**.
- The model achieved **97.2% validation accuracy** and **96.89% test accuracy**.

---

References
- **MNIST Dataset**: [MNIST Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)
- **Scikit-learn Documentation**: [scikit-learn.org](https://scikit-learn.org)
- **Random Forest Algorithm**: Breiman, L. (2001). "Random Forests". Machine Learning.

This README provides full instructions for running and understanding the MNIST classification project.

