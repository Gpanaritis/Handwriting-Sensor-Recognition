# Handwriting-Sensor-Recognition

## Introduction

In this report, we will discuss the process of converting the data and generating a model for handwriting sensor recognition. The goal of this project was to develop a model that can accurately identify different classes of handwriting based on sensor data.

## Installation

1. Please install `python` and `mongodb` on your device

2. Install pipenv
```
pip install pipenv
```
3. Cd to project folder and install the dependencies
```
pipenv install
```
4. Download the data
```
curl -L -o data.zip "https://drive.google.com/uc?export=download&id=1J0u1RfI5838teqmBPWm_GIldPL5gWPwX"
```
5. Extract the data
```
unzip data.zip
```

## Usage
1. In `config.yml` enter your mongodb uri

2. Don't forget to select the correct kernel on both `ipynb` files

1. Run all cells on `aiot_dataset_creation.ipynb`

2. Run all cells on `aiot_project_2023.ipynb`

## Report

### Data Conversion

1. Read data from the database and store it in a DataFrame.
2. Apply a sliding window technique to capture temporal information.
3. Use a low pass filter to remove high-frequency noise and smooth the data.
4. Flatten the 2D instances into a 1D vector and replace None values with NaN.
5. Split the data into training and testing sets, and scale the features.
6. Impute NaN values using custom weights to handle missing data.
7. Apply PCA to reduce the dimensionality of the data.

These steps were taken to convert the sensor data into a suitable format for further processing and model training.
### Model Generation

In the model generation phase, we experimented with different machine learning algorithms, including SVM, Random Forest, and CNN. For SVM and Random Forest, we utilized the data transformed by PCA, while CNN used the raw data without PCA.

We performed grid search to find the best hyperparameters for SVM and Random Forest models. Grid search involved testing various combinations of hyperparameters to determine the optimal configuration that yielded the highest accuracy.

In the case of CNN, we focused on exploring different architectures by varying the number of neurons in the network. This allowed us to evaluate the impact of network complexity on the model's accuracy.

By trying different algorithms and tuning their hyperparameters, we aimed to find the best model that could accurately classify the handwriting sensor data.

### Model Accuracy and Observations


| Model         | Best Parameters                               | Best estimator                                         | Best score         | Accuracy    |
|---------------|-----------------------------------------------|--------------------------------------------------------|--------------------|-------------|
| SVM           | {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}    | SVC(C=10, gamma=0.001)                                 | 0.8552036199095022 | 0.828125    |
| Random Forest | {'max_depth': 30, 'n_estimators': 200}        | RandomForestClassifier(max_depth=30, n_estimators=200) | 0.8316742081447964 | 0.890625    |
| CNN           | {'optimizer': 'adam', 'learning_rate': 0.001} | CNNModel(hidden_layer=[60])                            | N/A                | 0.876953125 |

Overall, the Random Forest model achieved the highest accuracy of 89.06%, closely followed by the CNN model with 87.70%. The SVM model also performed reasonably well with an accuracy of 82.81%. These results suggest that both ensemble methods (Random Forest) and deep learning (CNN) can be effective the classification task, but further analysis and evaluation of the CNN model are needed to obtain a comprehensive understanding of its performance.
### Classification Report

The following classification report provides detailed performance metrics for each letter class in the dataset (Result from the random forest):

Based on the provided classification report, here are the results for each letter class:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| α     | 0.71      | 0.62   | 0.67     | 8       |
| β     | 0.90      | 0.75   | 0.82     | 12      |
| γ     | 0.89      | 1.00   | 0.94     | 8       |
| δ     | 0.80      | 1.00   | 0.89     | 4       |
| ε     | 0.83      | 0.91   | 0.87     | 11      |
| ζ     | 1.00      | 0.89   | 0.94     | 9       |
| η     | 0.83      | 1.00   | 0.91     | 5       |
| θ     | 1.00      | 1.00   | 1.00     | 7       |

Observations:
- Overall, the model achieved high precision, recall, and F1-scores for most letter classes, indicating accurate predictions.
- Class 'α' had a relatively lower precision and recall compared to other classes, but still achieved a reasonable F1-score.
- Class 'β' had a lower recall compared to precision, indicating some difficulty in correctly identifying instances of this class.
- Classes 'γ', 'δ', 'ζ', 'η, and 'θ' achieved excellent precision, recall, and F1-scores, demonstrating high performance in classification.