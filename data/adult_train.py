# Import necessary libraries
import json
import os
import pickle
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data import load_config, create_folder_if_not_exists
from data.ml_utilities import label_encode_and_save_classes, construct_pipeline, train_model

DATASET_NAME = "adult"
config_path = f"./{DATASET_NAME}_model_config.json"
save_path = f"./{DATASET_NAME}"

# Mapping of categories to new integer representations
category_to_int = {
    'Primary Education': 1,
    'Middle School': 2,
    'High School': 3,
    'Undergraduate Level': 4,
    'Masters Level': 5,
    'Doctorate/Prof Level': 6
}


def map_capital_col(data, config):
    """Refine Investment Outcome with categories including Break-Even and No Activity."""
    # Initialize default category for all as 'No Activity (0$)'
    data['investmentoutcome'] = 'No Investment'

    # Calculate Investment Outcome for other cases
    investment_outcome = data['capitalgain'] - data['capitalloss']

    # Define bins and labels excluding the default 'No Activity' and 'Break-Even'
    bins = [-float('inf'), -1000, 0, 5000, float('inf')]
    labels = ['Major Loss (more than 1k$)', 'Minor Loss (up to 1k$)', 'Minor Gain (up to 5k$)',
              'Major Gain (above 5k$)']
    # Update categories based on bins and labels for non-default cases
    non_default_cases = investment_outcome != 0
    categorized_outcomes = pd.cut(investment_outcome[non_default_cases], bins=bins, labels=labels)
    data.loc[non_default_cases, 'investmentoutcome'] = categorized_outcomes

    # Updated mapping with new categories
    mapping = {
        "Major Loss (more than 1k$)": 0,
        "Minor Loss (up to 1k$)": 1,
        "No Investment": 2,
        "Minor Gain (up to 5k$)": 3,
        "Major Gain (above 5k$)": 4
    }

    # Add the new mapping to the config
    config["ordinal_mapping"]["investmentoutcome"] = mapping

    # Optionally, you may drop the original capital gain and loss columns
    data.drop(columns=['capitalgain', 'capitalloss'], inplace=True)


def map_education_to_int(education_str):
    """
    Maps an education string to its corresponding broad education category integer.

    Parameters:
    - education_str: A string representing the education level.

    Returns:
    - An integer representing the broad education category.
    """
    # Mapping of education levels to categories
    category_map = {
        'Preschool': 'Dropout',
        '1st-4th': 'Dropout',
        '5th-6th': 'Dropout',
        '7th-8th': 'Dropout',
        '9th': 'Dropout',
        '10th': 'Dropout',
        '11th': 'Dropout',
        '12th': 'Dropout',
        'HS-grad': 'High School grad',
        'Some-college': 'High School grad',
        'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates',
        'Bachelors': 'Bachelors grad',
        'Masters': 'Masters grad',
        'Prof-school': 'Professional Degree',
        'Doctorate': 'Doctorate/Prof Level'
    }

    # Get the category from the education level
    category = category_map.get(education_str, "Unknown")
    return category


def map_education_levels(data, config):
    # Apply binning to education.num
    def refined_bin_education_level(x):
        if x in [1, 2, 3, 4, 5, 6, 7, 8]:
            return 0  # Dropout
        elif x in [9, 10]:
            return 1  # High School grad
        elif x in [11, 12]:
            return 2  # Associates
        elif x == 13:
            return 3  # Bachelors
        elif x == 14:
            return 4  # Masters
        elif x == 15:
            return 5  # Professional Degree
        elif x == 16:
            return 6  # Doctorate/Prof Level
        else:
            return None  # For any value out of the original range

    data['educationlevel'] = data['educationnum'].apply(refined_bin_education_level)
    data.drop(['educationnum'], axis=1, inplace=True)

    refined_binned_category_map = {
        0: 'Primary Education',
        1: 'Middle School',
        2: 'High School without Graduation',
        3: 'High School Graduate',
        4: 'College without Degree',
        5: "Associate's Degrees",
        6: "Bachelor's Degree",
        7: 'Post-graduate Education'
    }

    # Reverse the mapping for the config
    refined_binned_category_map = {v: k for k, v in refined_binned_category_map.items()}

    # Add the mapping to the config
    config["ordinal_mapping"]["educationlevel"] = refined_binned_category_map


def standardize_column_names(data):
    # lower case column names and replace '.' with ''
    data.columns = data.columns.str.lower().str.replace('.', '')


def add_work_life_balance(data):
    categories = ['Poor', 'Fair', 'Good']
    mean = categories.index('Fair')  # Mean set to index of 'Fair' for Gaussian center
    std_dev = 0.75  # Standard deviation, adjust as needed for spread

    # Generate indices from a Gaussian distribution
    gaussian_indices = np.random.normal(loc=mean, scale=std_dev, size=len(data))

    # Clip the indices to lie within the range of categories to avoid out-of-bounds indices
    gaussian_indices_clipped = np.clip(gaussian_indices, 0, len(categories) - 1)

    # Round indices to nearest integer to use as valid category indices
    category_indices = np.round(gaussian_indices_clipped).astype(int)

    # Assign categories based on Gaussian-distributed indices
    data['worklifebalance'] = [categories[i] for i in category_indices]

    # Create a mapping from string values to integers
    mapping = {category: i for i, category in enumerate(categories)}
    # Apply the mapping to the 'WorkLifeBalance' column
    data['worklifebalance'] = data['worklifebalance'].map(mapping)
    return mapping


def fil_nans_with_mode(data):
    data[data == '?'] = np.nan  # Replace '?' with NaN
    cols_with_nan_names_list = data.columns[data.isna().any()].tolist()
    for col in cols_with_nan_names_list:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Check outliars in work hours (working with std dev)
    mean = data['hoursperweek'].mean()
    std = data['hoursperweek'].std()
    data_cleaned = data[(data['hoursperweek'] > mean - 2 * std) & (data['hoursperweek'] < mean + 2 * std)]
    print(f"Removed {len(data) - len(data_cleaned)} outliars from the data")
    return data_cleaned


def map_ordinal_columns(data, config):
    for col, mapping in config["ordinal_mapping"].items():
        if col in ["worklifebalance", "educationlevel"]:
            continue
        data[col] = data[col].map(mapping)


def bin_age_column(data):
    bins = [17, 25, 50, 100]
    labels = ["Young (18-25)", "Adult (25-50)", "Old (50-100)"]
    data["age"] = pd.cut(data["age"], bins=bins, labels=labels)


def bin_workclass_col(data):
    # Define the mapping from original labels to new categories
    workclass_map = {
        'Private': 'Private Sector',
        'State-gov': 'Government Employment',
        'Federal-gov': 'Government Employment',
        'Self-emp-not-inc': 'Self-Employed / Entrepreneurial',
        'Self-emp-inc': 'Private Sector',
        'Local-gov': 'Government Employment',
        'Without-pay': 'Unemployed / Other',
        'Never-worked': 'Unemployed / Other'
    }

    # Apply the mapping to the "Workclass" column
    data["workclass"] = data["workclass"].map(workclass_map)
    return data


def map_occupation_col(data):
    # Mapping of professions to their respective subgroups
    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
    }

    # Apply mapping to the occupation column
    data['occupation'] = data['occupation'].map(occupation_map)


def preprocess_marital_status(data):
    data["maritalstatus"] = data["maritalstatus"].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'],
                                                          'Single')
    data["maritalstatus"] = data["maritalstatus"].replace(
        ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')


def preprocess_data_specific(data, config):
    if "drop_columns" in config:
        data.drop(columns=config["drop_columns"], inplace=True)

    standardize_column_names(data)

    # Following functions assume capitalized col names
    config["ordinal_mapping"]['worklifebalance'] = add_work_life_balance(data)
    preprocess_marital_status(data)
    map_education_levels(data, config)
    map_capital_col(data, config)
    map_occupation_col(data)
    map_ordinal_columns(data, config)

    data = fil_nans_with_mode(data)

    # Check which column has "Other" values
    for col in data.columns:
        if "Other" in data[col].unique():
            print(f"Column '{col}' has 'Other' values.")

    target_col = config["target_col"]
    X = data.drop(columns=[target_col])
    y = data[target_col]

    if "rename_columns" in config:
        X = X.rename(columns=config["rename_columns"])

    if "columns_to_encode" in config:
        X, encoded_classes = label_encode_and_save_classes(X, config)
    else:
        encoded_classes = {}

    return X, y, encoded_classes


def main():
    config = load_config(config_path)
    config["save_path"] = save_path

    data = pd.read_csv(config["dataset_path"])
    create_folder_if_not_exists(save_path)

    X, y, encoded_classes = preprocess_data_specific(data, config)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    target_col = config["target_col"]
    # Add labels to X and save train and test data
    X_train[target_col] = y_train
    X_test[target_col] = y_test
    X_train.to_csv(os.path.join(save_path, f"{DATASET_NAME}_train.csv"))
    X_test.to_csv(os.path.join(save_path, f"{DATASET_NAME}_test.csv"))
    X_train.drop(columns=[target_col], inplace=True)
    X_test.drop(columns=[target_col], inplace=True)

    # Check if there are nan values in the data and raise an error if there are
    if X_train.isna().sum().sum() > 0:
        print(X_train.isna().sum())
        raise ValueError("There are NaN values in the training data.")

    # Copy the config file to the save path
    with open(os.path.join(save_path, f"{DATASET_NAME}_model_config.json"), 'w') as file:
        json.dump(config, file)

    # Change list of column names to be encoded to a list of column indices
    columns_to_encode = [X_train.columns.get_loc(col) for col in config["columns_to_encode"]]
    pipeline = construct_pipeline(columns_to_encode, RandomForestClassifier())

    model_params = {("model__" + key if not key.startswith("model__") else key): value for key, value in
                    config["model_params"].items()}
    search_params = config.get("random_search_params", {"n_iter": 10, "cv": 5, "random_state": 42})

    best_model, best_params = train_model(X_train, y_train, pipeline, model_params, search_params)

    best_model.fit(X_train, y_train)

    # Predict probabilities for the train set
    y_train_pred = best_model.predict_proba(X_train)[:, 1]

    # Compute ROC AUC score for the train set
    train_score = roc_auc_score(y_train, y_train_pred)
    print("Best Model Score Train:", train_score)

    # Predict probabilities for the test set
    y_test_pred = best_model.predict_proba(X_test)[:, 1]

    # Compute ROC AUC score for the test set
    test_score = roc_auc_score(y_test, y_test_pred)
    print("Best Model Score Test:", test_score)

    """# Print evaluation metrics on train and test
    print("Best Model Score Train:", best_model.score(X_train, y_train, metric="roc_auc"))
    print("Best Model Score Test:", best_model.score(X_test, y_test,  metric="roc_auc"))
    print("Best Parameters:", best_params)"""

    # Save the best model
    pickle.dump(best_model, open(os.path.join(save_path, f"{DATASET_NAME}_model_rf.pkl"), 'wb'))


if __name__ == "__main__":
    main()
