# BeanVision – Dry Bean Classification Using Machine Learning

#### Problem Statement

The objective of this project is to classify dry beans into seven categories using their physical measurements, such as area, perimeter, axis lengths, compactness, and other shape-related features.

Five machine learning algorithms are implemented and compared: Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and SVM.

Their performance is evaluated using Accuracy, AUC, Precision, Recall, F1-score, and MCC. The trained models are also integrated into a Streamlit application, allowing users to upload data, select a model, and view predictions and evaluation results.

## 1. About the Project

For this project, I worked on classifying different varieties of dry beans using machine learning. The aim was to train multiple classification models using the physical measurements of beans and compare how well they identify each variety.

I selected the Dry Bean Dataset from the UCI Machine Learning Repository. It contains 16 numerical features describing the shape and size of dry beans. The target column contains seven different bean varieties.

I implemented five classification algorithms, evaluated their predictions using six performance metrics, and developed a Streamlit application to make the results easier to explore.

The application uses the models saved after training. A user can upload a test dataset, select a model and view its evaluation results without training the model again.

## 2. Dataset Used

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The dataset contains measurements collected from seven varieties of dry beans.

### Dataset summary

| Detail | Value |
|---|---:|
| Original number of records | 13,611 |
| Records after removing duplicates | 13,543 |
| Number of input features | 16 |
| Target column | Class |
| Number of bean varieties | 7 |
| Missing values | 0 |
| Duplicate records removed | 68 |

### Bean varieties

The classification target contains these seven labels:

- SEKER
- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SIRA

### Input features

The following 16 numerical columns were used as input to the models:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRatio
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. Roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

The column `Class` was used as the target variable.

## 3. Data Preparation

I first loaded the dataset into a Pandas DataFrame and checked its shape, column information, data types and missing values.

There were no missing values, but I found 68 duplicate records. I removed these records before proceeding with the analysis. The resulting dataset contained 13,543 records.

After cleaning the data, I separated the 16 input features from the target column.

I then divided the dataset into training and testing sets using an 80:20 split. Stratification was used so that the distribution of the seven bean varieties remained approximately consistent in both sets.

### Train-test split results

| Set | Records |
|---|---:|
| Training data | 10,834 |
| Testing data | 2,709 |
| Total | 13,543 |

The random state used for the split was 42.

### Feature scaling

StandardScaler was used for Logistic Regression, KNN and SVM because these algorithms can be affected by differences in feature scales.

Decision Tree and Gaussian Naive Bayes were trained without feature scaling.

## 4. Exploratory Data Analysis

Before training the models, I explored the dataset to understand the distribution of the bean varieties and the relationships between the numerical features.

The following plots were generated during the analysis:

**Bean variety distribution:** This plot shows how many records belong to each of the seven classes. It also helps identify differences in class representation.

**Correlation heatmap:** I used this plot to examine the relationships between numerical features and identify features that are strongly correlated.

**Area boxplot:** This plot compares the area measurements across the different bean varieties and helps examine their distributions and possible outliers.

## 5. Classification Models

I trained five classification algorithms using the same training and testing split.

### Logistic Regression

Logistic Regression was included as one of the classification methods for the experiment. It estimates class probabilities using the input features and assigns a predicted class.

### Decision Tree

The Decision Tree model learns a set of decision rules from the training data. It uses these rules to classify new observations.

### K-Nearest Neighbors (KNN)

KNN predicts the class of an observation by looking at its nearest training examples. Feature scaling was applied before using this algorithm.

### Gaussian Naive Bayes

Gaussian Naive Bayes is based on Bayes' theorem. It estimates class probabilities while assuming that the numerical features follow Gaussian distributions within each class.

### Support Vector Machine (SVM)

SVM was used to separate the bean varieties by learning classification boundaries. Probability estimates were enabled so that the model could also be evaluated using multiclass AUC.

## 6. Evaluation Metrics

All five trained models were evaluated using the same test dataset of 2,709 records.

I used the following six metrics to compare their performance:

| Metric | What it tells us |
|---|---|
| Accuracy | How many test records were classified correctly. |
| AUC | How well the model distinguishes between the classes. |
| Precision | How reliable the model's positive class predictions are. |
| Recall | How many actual class instances the model identifies correctly. |
| F1 Score | A combined measure of precision and recall. |
| MCC | An overall measure of agreement between actual and predicted labels. |

Since this is a multiclass classification problem, macro averaging was used for Precision, Recall and F1 Score. AUC was calculated using the One-vs-Rest approach with macro averaging.

## 7. Model Evaluation Results

The following table contains the results obtained from the test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9947 | 0.9311 | 0.9303 | 0.9306 | 0.9028 |
| Decision Tree | 0.8955 | 0.9460 | 0.9094 | 0.9104 | 0.9098 | 0.8737 |
| KNN | 0.9155 | 0.9832 | 0.9290 | 0.9256 | 0.9270 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| SVM | **0.9210** | 0.9945 | **0.9342** | **0.9317** | **0.9328** | **0.9045** |

### Observations

From the results, I observed the following:

- SVM achieved the highest accuracy of 92.10%.
- SVM also obtained the highest Precision, Recall, F1 Score and MCC.
- Logistic Regression recorded the highest AUC of 0.9947.
- KNN performed close to Logistic Regression and SVM.
- Decision Tree gave an accuracy of 89.55%.
- Gaussian Naive Bayes had the lowest accuracy among the five models.

Based on these results, SVM gave the best overall performance across most of the metrics. However, Logistic Regression had a slightly higher AUC. The difference between their AUC values was small.

## 8. Streamlit Application – BeanVision

After training and evaluating the models, I developed a Streamlit application called **BeanVision – ML Classification Studio**.

The purpose of the application is to make the trained models available through a simple interface.

### Features available in the application

**Upload dataset**

The user can upload a CSV file containing all 16 input features and the actual `Class` column.

**Select a model**

The application provides five model choices: Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM.

**View dataset information**

The uploaded dataset's record count, feature count and number of bean varieties are displayed. A preview of the uploaded data is also available.

**View evaluation metrics**

After selecting a model, the application displays Accuracy, AUC, Precision, Recall, F1 Score and MCC.

**Confusion matrix and classification report**

The confusion matrix shows the actual and predicted bean varieties. The classification report provides class-wise performance details.

**Compare models**

The comparison section displays the performance of all five models using the uploaded test dataset. The user can select a metric to compare their results.

**Dataset insights**

The application displays the distribution of bean varieties in the uploaded dataset.

**Download predictions**

The user can download a CSV containing the uploaded records along with the predicted bean variety.

### How the application works

The five trained models were saved using Joblib. When the Streamlit application starts, it loads these saved model files.

When a user uploads a CSV, the application checks the required columns and passes the input features to the selected model. It then compares the predictions with the actual `Class` values to calculate the evaluation metrics.

The application does not perform model training again during this process.

## 9. Project Folder Structure

The project is organised as follows:

```text
DryBean_ML_assignment/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── feature_columns.json
```

### Files in the project

| File | Purpose |
|---|---|
| `app.py` | Contains the Streamlit application code. |
| `requirements.txt` | Lists the Python libraries required to run the application. |
| `README.md` | Provides project information and execution instructions. |
| `test_data.csv` | Contains the test records and actual bean variety labels. |
| `models/` | Stores the trained machine learning models. |
| `feature_columns.json` | Stores the names and order of the input features. |

## 10. Tools and Libraries

The following tools and libraries were used:

- **Python:** Main programming language.
- **Pandas:** Loading, cleaning and handling the dataset.
- **NumPy:** Numerical operations.
- **Scikit-learn:** Model training, preprocessing and evaluation.
- **Matplotlib:** Plotting graphs.
- **Seaborn:** Creating statistical visualisations.
- **Joblib:** Saving and loading trained models.
- **Streamlit:** Building the web application.
- **GitHub:** Hosting the project source code.
- **Streamlit Community Cloud:** Hosting the web application.

## 11. Running the Application

Follow these steps to run BeanVision locally.

### Step 1: Clone the repository

```bash
git clone https://github.com/2026aj05371/ML_Assignment_BeanVision-Dry-Bean-Classification.git
```

### Step 2: Open the project folder

```bash
cd DryBean_Ml_assignment
```

### Step 3: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 4: Start the application

```bash
streamlit run app.py
```

### Step 5: Open the application

Open the local URL displayed in the terminal. The default address is:

```text
http://localhost:8501
```

Upload a compatible test CSV and select a model to view its results.

## 12. Input File Requirements

For evaluation, the uploaded CSV must contain:

- All 16 numerical input features.
- A target column named `Class`.

The target column is needed because the application calculates evaluation metrics by comparing actual labels with predicted labels.

The uploaded data should use the same feature names and compatible feature values as the data used during model training.

## 13. Conclusion

In this assignment, I used the Dry Bean Dataset to build and compare five machine learning classification models.

The models produced different results on the test dataset. SVM achieved the highest accuracy and the strongest overall performance across most of the evaluation metrics. Logistic Regression recorded the highest AUC, while KNN also performed well.

I also developed the BeanVision Streamlit application to make it possible to evaluate the saved models using an uploaded test dataset.

This work helped me understand the complete classification workflow, from preparing the dataset and training models to evaluating their performance and presenting the results through a web application.

## 14. Project Details

**Project:** Dry Bean Variety Classification Using Machine Learning  
**Application:** BeanVision – ML Classification Studio  
**Programme:** M.Tech in Artificial Intelligence and Machine Learning  
**Institution:** BITS Pilani – WILP  
**BITS ID:** 2026AJ05371  
**Assignment:** Machine Learning Assignment 2  
**Dataset:** UCI Dry Bean Dataset

## 15. Project Links

**GitHub Repository:** https://github.com/2026aj05371/ML_Assignment_BeanVision-Dry-Bean-Classification.git

**Live Streamlit Application:** 

## 16. Dataset Acknowledgement

The dataset used for this project was obtained from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/602/dry+bean+dataset