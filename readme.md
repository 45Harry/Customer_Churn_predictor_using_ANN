# Customer Churn Prediction Using Neural Networks

## About

This project focuses on predicting customer churn (whether a customer will leave or stay with the company) using a deep learning model. The dataset contains various features about customers, including demographic details, account information, and service usage. The goal is to train a neural network model to classify whether a customer will churn (Exited = 1) or not (Exited = 0).

The model uses a fully connected neural network (also known as a dense network) built with TensorFlow and Keras. The project also includes data preprocessing, exploratory data analysis (EDA), and evaluation using various metrics like accuracy, F1-score, and confusion matrix.

## Dataset

The dataset used for this project is `Churn_Modelling.csv`, which contains the following columns:

- `RowNumber`: Row number in the dataset (not relevant for prediction)
- `CustomerId`: Unique customer ID (not relevant for prediction)
- `Surname`: Customer's surname (not relevant for prediction)
- `CreditScore`: Credit score of the customer
- `Geography`: The country where the customer is located (France, Germany, Spain)
- `Gender`: Gender of the customer (Male/Female)
- `Age`: Age of the customer
- `Tenure`: Number of years the customer has been with the company
- `Balance`: Customer's account balance
- `NumOfProducts`: Number of products the customer has purchased
- `HasCrCard`: Whether the customer has a credit card (1 = Yes, 0 = No)
- `IsActiveMember`: Whether the customer is an active member (1 = Yes, 0 = No)
- `EstimatedSalary`: Estimated annual salary of the customer
- `Exited`: Target variable (1 = Churned, 0 = Stayed)

## Features

- **Data Preprocessing**: Cleaned the dataset by removing irrelevant columns, handling categorical features (`Gender`, `Geography`), and scaling continuous variables (`Balance`, `EstimatedSalary`).
- **Exploratory Data Analysis (EDA)**: Visualized customer tenure and salary distributions to gain insights into how they relate to churn.
- **Model**: Built a fully connected neural network model with two hidden layers using TensorFlow/Keras.
- **Model Evaluation**: Evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix.

## Project Structure

```plaintext
.
├── Churn_Modelling.csv       # The dataset used for training
├── churn_prediction_model.py # Python script with data preprocessing, model creation, and evaluation
└── README.md                 # This README file
```

## Requirements

To run this project, you will need to install the following libraries:

- Python 3.x
- TensorFlow
- pandas
- matplotlib
- scikit-learn
- seaborn

You can install the dependencies using pip:

```bash
pip install tensorflow pandas matplotlib scikit-learn seaborn
```

## Usage

### Step 1: Load the dataset
The `Churn_Modelling.csv` dataset should be available in the same directory as the script.

### Step 2: Preprocess the data
The dataset is cleaned by:
- Removing irrelevant columns.
- Encoding categorical columns (`Gender`, `Geography`).
- Scaling numerical features.

### Step 3: Split the data
The dataset is split into training and test sets, with a stratified split to ensure equal distribution of the target variable (`Exited`).

### Step 4: Train the model
A deep neural network model is built using TensorFlow's Keras API, with the following architecture:
- Input layer (12 features)
- 2 hidden layers (100 units each, ReLU activation, Batch Normalization)
- Output layer (1 unit, sigmoid activation)

The model is compiled using the `adam` optimizer and `binary_crossentropy` loss function.

### Step 5: Model Evaluation
The model is trained for 100 epochs. After training, the model's performance is evaluated on the test set using metrics like accuracy, precision, recall, F1-score, and the confusion matrix.

### Step 6: Visualization
A confusion matrix heatmap is displayed using Seaborn to visualize the true vs. predicted churn values.

## Results

After training the model, the performance metrics (accuracy, precision, recall, and F1-score) provide a clear understanding of how well the model is performing. The confusion matrix also highlights the misclassifications made by the model (false positives and false negatives).

## Potential Improvements

- **Handling Class Imbalance**: Use techniques such as oversampling the minority class or adjusting class weights to handle the class imbalance in the dataset.
- **Hyperparameter Tuning**: Perform grid search or random search for hyperparameter optimization to improve model performance.
- **Alternative Models**: Try other machine learning models (e.g., Random Forest, XGBoost, Logistic Regression) and compare performance.

## License

This project is licensed under the MIT License.

---
