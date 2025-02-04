# ml_utilities.py
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
import os


def label_encode_and_save_classes(df, config):
    def sort_keys_by_values(mapping_dict):
        # Sort the dictionary by values and return the keys as a list
        return [key for key, value in sorted(mapping_dict.items(), key=lambda item: item[1])]

    encoded_classes = {}
    categorical_mapping = {}
    columns = config["columns_to_encode"]
    save_path = config["save_path"]
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        # Create a dictionary mapping for each column
        mapping_dict = {int(i): str(class_name) for i, class_name in enumerate(le.classes_)}
        encoded_classes[col] = mapping_dict
        # categorical_mapping: map from integer to list of strings, names for each
        # value of the categorical features.
        col_id = df.columns.get_loc(col)
        categorical_mapping[col_id] = list(mapping_dict.values())

    # Enrich both mappings with mappings from config "ordinal_mapping"
    ordinal_mapping = config.get("ordinal_mapping", {})
    # Remove Target column from ordinal_mapping
    ordinal_mapping.pop(config["target_col"], None)
    # Add the ordinal mappings to the encoded_classes
    for col, mapping in ordinal_mapping.items():
        col_id = df.columns.get_loc(col)
        # turn mapping to a list of strings in the same order as the integers
        mapping_list = sort_keys_by_values(mapping)
        categorical_mapping[str(col_id)] = mapping_list
        encoded_classes[col] = {int(i): str(class_name) for i, class_name in enumerate(mapping)}

    # Save all the mappings in separate JSON files
    with open(os.path.join(save_path, "encoded_col_mapping.json"), 'w') as f:
        json.dump(encoded_classes, f)
    with open(os.path.join(save_path, "categorical_mapping.json"), 'w') as f:
        json.dump(categorical_mapping, f)

    return df, encoded_classes


def preprocess_data(data, columns_to_encode, ordinal_info, drop_columns=None):
    if drop_columns:
        data.drop(columns=drop_columns, inplace=True)
    if columns_to_encode:
        data, encoded_classes = label_encode_and_save_classes(data, columns_to_encode)
    if ordinal_info:
        for col, mapping in ordinal_info.items():
            data[col] = data[col].map(mapping)
    return data


def construct_pipeline(columns_to_encode, model):
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(handle_unknown='ignore'), columns_to_encode),
        ], remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('to_array', FunctionTransformer(np.asarray, validate=False)),
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    return pipeline


def train_model(X, y, pipeline, model_params, search_params):
    random_search = RandomizedSearchCV(pipeline, param_distributions=model_params, **search_params)
    random_search.fit(X, y)
    return random_search.best_estimator_, random_search.best_params_
