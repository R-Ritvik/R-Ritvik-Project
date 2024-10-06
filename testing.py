import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

@pytest.fixture()
def data_split():
    df2 = pd.read_csv('cleaned_data.csv')
    sc = MinMaxScaler()
    df2['tenure'] = sc.fit_transform(df2[['tenure']])
    df2['MonthlyCharges'] = sc.fit_transform(df2[['MonthlyCharges']])
    df2['TotalCharges'] = sc.fit_transform(df2[['TotalCharges']])
    X = df2.drop(['Churn', 'customerID'], axis=1)
    y = df2['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def test_data_loading():
    df1 = pd.read_csv('telecom_customer_churn.csv')

    assert not df1.empty, "DataFrame is empty!"

    expected_columns = ['gender', 'SeniorCitizen', 'Churn']
    for col in expected_columns:
        assert col in df1.columns, f"Missing expected column: {col}"

# Test function to check for missing values
def test_missing_values():
    df = pd.read_csv('cleaned_data.csv')
    
    assert df.isnull().sum().sum() == 0, "There are missing values in the dataset"


# Test function to check data types of specific columns
def test_data_types():
    df = pd.read_csv('cleaned_data.csv')
    
    assert df['tenure'].dtype == 'int64', "tenure is not an int"
    assert df['MonthlyCharges'].dtype == 'float64', "MonthlyCharges is not a float"
    assert df['TotalCharges'].dtype == 'float64', "TotalCharges is not a float"


# Test function for checking feature scaling
def test_feature_scaling(data_split):
    X_train, _, _, _ = data_split
    assert (X_train['tenure'].min() >= 0) and (X_train['tenure'].max() <= 1), \
        "tenure is not scaled between 0 and 1"
    assert (X_train['MonthlyCharges'].min() >= 0) and (X_train['MonthlyCharges'].max() <= 1), \
        "MonthlyCharges is not scaled between 0 and 1"
    assert (X_train['TotalCharges'].min() >= 0) and (X_train['TotalCharges'].max() <= 1), \
        "TotalCharges is not scaled between 0 and 1"


# Test function for Logistic Regression
def test_logistic_regression(data_split):
    X_train, X_test, y_train, y_test = data_split
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.7, f"Expected accuracy > 0.7, but got {accuracy}"

# Test function for Random Forest
def test_random_forest(data_split):
    X_train, X_test, y_train, y_test = data_split
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.7, f"Expected accuracy > 0.7, but got {accuracy}"

# Test function for Decision Tree
def test_decision_tree(data_split):
    X_train, X_test, y_train, y_test = data_split
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.7, f"Expected accuracy > 0.7, but got {accuracy}"

# Test function for Support Vector Machine
def test_svm(data_split):
    X_train, X_test, y_train, y_test = data_split
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.7, f"Expected accuracy > 0.7, but got {accuracy}"

# Test function for Naive Bayes
def test_naive_bayes(data_split):
    X_train, X_test, y_train, y_test = data_split
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.7, f"Expected accuracy > 0.7, but got {accuracy}"