
# 🧠 Diabetes Prediction Model – Project Documentation

This project builds a machine learning model to predict diabetes based on health-related features using Support Vector Machine (SVM). The workflow follows a step-by-step process including data handling, visualization, modeling, and persistence using object-oriented programming.

---

## Step 1: 📥 Data Collection & 🔍 Understanding

- **Objective**: Load the dataset and understand its structure.
- **Implementation**:
  - Used a `DataHandler` class to load the CSV file and explore:
    - Shape of the dataset
    - Info on data types
    - Descriptive statistics
    - Null values

---

## Step 2: 🧹 Data Preprocessing & 📊 Analysis

### Data Preprocessing
- **Objective**: Prepare the data for modeling.
- **Implementation**:
  - Created a `Preprocessor` class to fill missing values with column means.

### Univariate Analysis
- **Objective**: Explore the distribution of each feature.
- **Implementation**:
  - `UnivariateAnalysis` class plots histograms and KDEs for numeric features.

### Bivariate Analysis
- **Objective**: Study relationships between features.
- **Implementation**:
  - `BivariateAnalysis` class visualizes correlations using a heatmap to understand how features relate to each other and to the target variable (`diabetes`).

---

## Step 3: ✂ Data Splitting

- **Objective**: Split the dataset into training and testing sets.
- **Implementation**:
  - `DataSplitter` class separates features and target, then uses `train_test_split()` to divide the data.

---

## Step 4: 🤖 Model Training

- **Objective**: Train a machine learning model to predict diabetes.
- **Implementation**:
  - `SVMModel` class uses a Support Vector Classifier (`SVC`) from `sklearn`.
  - Model is trained on the training data.
  - Evaluation is performed on the test set with:
    - Accuracy score
    - Confusion matrix
    - Classification report

---

## Step 5: 💾 Model Storage

- **Objective**: Save the trained model for future use.
- **Implementation**:
  - `ModelSaver` class serializes the trained SVM model using Python’s `pickle` module.
