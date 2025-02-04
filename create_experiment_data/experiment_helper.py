import decimal
import json
from typing import List
import gin

import numpy as np
import pandas as pd
import copy

from data.response_templates.template_manager import TemplateManager


@gin.configurable
class ExperimentHelper:
    def __init__(self, conversation,
                 categorical_features,
                 feature_ordering=None,
                 categorical_mapping_path=None,
                 encoded_col_mapping_path=None,
                 actionable_features=None):
        self.conversation = conversation
        self.categorical_mapping_path = categorical_mapping_path
        self.encoded_col_mapping_path = encoded_col_mapping_path
        self.categorical_mapping = None
        self.categorical_features = categorical_features
        self.template_manager = None
        self.instances = {"train": [], "test": {}}
        self.current_instance = None
        self.current_instance_type = None
        self.instance_counters = {"train": 0, "test": 0, "final_test": 0, "intro_test": 0}
        self.feature_ordering = feature_ordering
        self.categorical_mapping = self.load_mapping(self.categorical_mapping_path)
        self.load_template_manager()
        self.actionable_features = actionable_features if actionable_features is not None else list()

    def load_instances(self):
        self._load_data_instances()
        self._load_test_instances()

    def load_mapping(self, path):
        # Load categorical mapping
        if path is not None:
            with open(path, "r") as f:
                mapping = json.load(f)
                mapping = {int(k): v for k, v in mapping.items()}
        else:
            mapping = None
        return mapping

    def load_template_manager(self):
        # Load Template Manager
        template_manager = TemplateManager(self.conversation,
                                           encoded_col_mapping_path=self.encoded_col_mapping_path,
                                           categorical_mapping=self.categorical_mapping)
        self.template_manager = template_manager

    def _load_data_instances(self):
        diverse_instances = self.conversation.get_var("diverse_instances").contents
        self.instances["train"] = [self._prepare_instance_data(instance) for instance in diverse_instances]

    def _load_test_instances(self):
        test_instances = self.conversation.get_var("test_instances").contents
        self.instances["test"] = {instance_id: self._process_test_instances(instances_dict)
                                  for instance_id, instances_dict in test_instances.items()}

    def _convert_values_to_string(self, instance):
        for key, value in instance.items():
            # Check if the value is a dictionary (to handle 'current' and 'old' values)
            if isinstance(value, dict):
                # Iterate through the inner dictionary and convert its values to strings
                for inner_key, inner_value in value.items():
                    # Turn floats to strings, converting to int first if no decimal part
                    if isinstance(inner_value, float) and inner_value.is_integer():
                        inner_value = int(inner_value)
                    value[inner_key] = str(inner_value)
            else:
                # Handle non-dictionary values as before
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                instance[key] = str(value)

    def _make_displayable_instance(self, instance, return_probabilities=False):
        # Round instance features
        instance = copy.deepcopy(instance)
        self._round_instance_features(instance[1])
        self.template_manager.apply_categorical_mapping(instance[1])
        # Order instance features and values according to the feature ordering
        if self.feature_ordering is not None:
            # Order instance features according to the feature ordering
            instance_features = instance[1]
            instance_features = {feature: instance_features[feature] for feature in self.feature_ordering}
            instance = (instance[0], instance_features, instance[2], instance[3], instance[4])
        else:  # alphabetically order features
            instance = (instance[0], dict(sorted(instance[1].items())), instance[2], instance[3], instance[4])
        # Make sure all values are strings
        self._convert_values_to_string(instance[1])
        # Make display feature names for the instance keys
        new_instance = self.template_manager.replace_feature_names_by_display_names(instance[1])
        # If not return probability and probabilities are not none, turn to class label
        if not return_probabilities and instance[2] is not None:
            ml_prediction = np.argmax(instance[2])
            display_ml_prediction = self.conversation.class_names[ml_prediction]
            instance = (instance[0], new_instance, display_ml_prediction, instance[3], instance[4])
        else:
            instance = (instance[0], new_instance, instance[2], instance[3], instance[4])
        return instance

    def get_next_instance(self, instance_type, return_probability=False):
        old_instance = None
        load_instance_methods = {"train": self._load_data_instances, "test": self._load_test_instances}
        get_instance_methods = {
            "train": lambda: self._get_training_instance(return_probability),
            "test": lambda: self._get_test_instance(self.current_instance[0] if self.current_instance else None),
            "final_test": self._get_final_test_instance,
            "intro_test": self._get_intro_test_instance
        }

        if not self.instances.get(instance_type, []):
            load_instance_methods.get(instance_type, lambda: None)()

        if instance_type != "train":
            instance_id, instance, counter = get_instance_methods[instance_type]()
            old_instance = self.current_instance
            predicted_label_index = np.argmax(
                self.conversation.get_var("model_prob_predict").contents(pd.DataFrame(instance, index=[0])))
            model_predicted_label = self.conversation.class_names[predicted_label_index]
            probability = None
            instance = (instance_id, instance, probability, model_predicted_label, predicted_label_index)
        else:  # "train"
            instance, counter, probability = get_instance_methods[instance_type]()
            predicted_label_index = np.argmax(
                self.conversation.get_var("model_prob_predict").contents(pd.DataFrame(instance[1], index=[0])))
            instance = (instance[0], instance[1], instance[2], instance[3], predicted_label_index)

        self.current_instance_type = instance_type
        instance = self._make_displayable_instance(instance)

        if old_instance and not instance_type in ["final_test", "intro_test"]:
            for key, value in instance[1].items():
                if value != old_instance[1][key]:
                    instance[1][key] = {"old": old_instance[1][key], "current": value}
        self.current_instance = instance
        return self.current_instance, counter, self.current_instance_type

    def _prepare_instance_data(self, instance):
        # Simplified example of preparing a data instance
        model_prediction = \
            self.conversation.get_var("model_prob_predict").contents(pd.DataFrame(instance['values'], index=[0]))[0]
        true_label = self._fetch_true_label(instance['id'])
        return instance['id'], instance['values'], model_prediction, true_label

    def _process_test_instances(self, instances_dict):
        # Example process for test instances
        return {comp: pd.DataFrame(data).to_dict('records')[0] for comp, data in instances_dict.items()}

    def _get_training_instance(self, return_probability, instance_type="train"):
        if not self.instances[instance_type]:
            return None, self.instance_counters[instance_type]
        self.current_instance = self.instances[instance_type][self.instance_counters[instance_type]]
        # Increment the counter for the next call.
        self.instance_counters[instance_type] += 1
        return self.current_instance, self.instance_counters[instance_type], self.current_instance[2]

    def _get_test_instance(self, train_instance_id, instance_type="test"):
        if not self.instances[instance_type]:
            return None, self.instance_counters[instance_type]

        instance_key = "least_complex_instance" if self.instance_counters[
                                                       instance_type] % 2 == 0 else "easy_counterfactual_instance"
        test_instances_dict = self.instances[instance_type][train_instance_id]
        instance = test_instances_dict[instance_key]
        self.instance_counters[instance_type] += 1
        return train_instance_id, instance, self.instance_counters[instance_type]

    def _get_final_test_instance(self, instance_type="final_test"):
        if not self.instances["test"]:
            return None, self.instance_counters["test"]

        instance_key = "most_complex_instance"
        # Get final test instance based on train instance id
        train_instance_id = self.instances["train"][self.instance_counters[instance_type]][0]
        test_instances_dict = self.instances["test"][train_instance_id]
        instance = test_instances_dict[instance_key]
        self.instance_counters[instance_type] += 1
        return train_instance_id, instance, self.instance_counters[instance_type]

    def _get_intro_test_instance(self, instance_type="intro_test"):
        if not self.instances["test"]:
            # Load test instances if not already loaded
            self._load_test_instances()
        if not self.instances["train"]:
            # Load training instances if not already loaded
            self._load_data_instances()

        instance_key = "most_complex_instance"  # (Dimi) Same as final test for now...
        # Get intro test instance based on train instance id
        train_instance_id = self.instances["train"][self.instance_counters[instance_type]][0]
        test_instances_dict = self.instances["test"][train_instance_id]
        instance = test_instances_dict[instance_key]
        self.instance_counters[instance_type] += 1
        return train_instance_id, instance, self.instance_counters[instance_type]

    def _round_instance_features(self, features):
        for feature, value in features.items():
            if isinstance(value, float):
                features[feature] = round(value, self.conversation.rounding_precision)

    def _fetch_true_label(self, instance_id):
        true_label = self.conversation.get_var("dataset").contents['y'].loc[instance_id]
        return self.conversation.class_names[true_label]

    def get_counterfactual_instances(self,
                                     original_instance):
        try:
            original_instance_id = original_instance.index[0]
        except IndexError:
            raise IndexError("Original instance is empty.")

        dice_tabular = self.conversation.get_var('tabular_dice').contents
        # Turn original intstance into a dataframe
        if not isinstance(original_instance, pd.DataFrame):
            original_instance = pd.DataFrame.from_dict(original_instance["values"], orient="index").transpose()
        cfes = dice_tabular.run_explanation(original_instance, "opposite", limit_features_to_vary=False)
        final_cfs_df = cfes[original_instance_id].cf_examples_list[0].final_cfs_df
        if final_cfs_df is None:
            return None
        # drop y column if it exists
        if "y" in final_cfs_df.columns:
            final_cfs_df = final_cfs_df.drop(columns=["y"])
        return final_cfs_df

    def get_similar_instance(self,
                             original_instance,
                             model,
                             max_features_to_vary=2):
        result_instance = None
        changed_features = 0
        for feature_name in self.actionable_features:
            # randomly decide if this feature should be changed
            if np.random.randint(0, 3) == 0:  # 66% chance to change
                continue
            tmp_instance = original_instance.copy() if result_instance is None else result_instance.copy()

            # Get random change value for this feature
            if self.categorical_features is not None and feature_name in self.categorical_features:
                try:
                    max_feature_value = len(
                        self.categorical_mapping[original_instance.columns.get_loc(feature_name)])
                except KeyError:
                    raise KeyError(f"Feature {feature_name} is not in the categorical mapping.")
                random_change = np.random.randint(1, max_feature_value)
                tmp_instance.at[tmp_instance.index[0], feature_name] += random_change
                tmp_instance.at[tmp_instance.index[0], feature_name] %= max_feature_value
            else:
                # Sample around mean for numerical features
                feature_mean = np.mean(self.conversation.get_var('dataset').contents["X"][feature_name])
                feature_std = np.std(self.conversation.get_var('dataset').contents["X"][feature_name])
                feature_min = np.min(self.conversation.get_var('dataset').contents["X"][feature_name])
                feature_max = np.max(self.conversation.get_var('dataset').contents["X"][feature_name])
                # Sample around feature mean
                random_change = np.random.normal(loc=feature_mean, scale=feature_std)
                while random_change == tmp_instance.at[tmp_instance.index[0], feature_name]:
                    random_change = np.random.normal(loc=feature_mean, scale=feature_std)
                # Check if the new value is within the feature range
                if random_change < feature_min:
                    random_change = feature_min
                elif random_change > feature_max:
                    random_change = feature_max

                # Check precision of the feature and round accordingly
                # Sample feature values to determine precision
                feature_values = self.conversation.get_var('dataset').contents["X"][feature_name]

                # Check if the feature values are integers or floats
                if pd.api.types.is_integer_dtype(feature_values):
                    precision = 0  # No decimal places for integers
                elif pd.api.types.is_float_dtype(feature_values):
                    # Determine the typical number of decimal places if floats
                    # This could be based on the standard deviation, min, max, or a sample of values
                    decimal_places = np.mean([abs(decimal.Decimal(str(v)).as_tuple().exponent) for v in
                                              feature_values.sample(min(100, len(feature_values)))])
                    precision = int(decimal_places)
                else:
                    # Default or handle other types as necessary
                    precision = 2  # Default to 2 decimal places for non-numeric types or as a fallback

                random_change = round(random_change, precision)

                tmp_instance.at[tmp_instance.index[0], feature_name] = random_change

            # After modifying the feature, check if it has actually changed compared to the original_instance
            if not tmp_instance.at[tmp_instance.index[0], feature_name] == original_instance.at[
                original_instance.index[0], feature_name]:
                result_instance = tmp_instance.copy()
                changed_features += 1  # Increment only if the feature has actually changed
            else:
                continue  # Skip to the next feature if no change was detected

            # Removed the redundant increment of changed_features and assignment of result_instance here
            if changed_features == max_features_to_vary:
                break

        return result_instance, changed_features

    def get_feature_names(self):
        feature_display_names = self.template_manager.feature_display_names.feature_name_to_display_name
        original_feature_names = list(self.conversation.get_var("dataset").contents['X'].columns)

        # If "unnamed: 0" is in the feature names, remove it
        if "unnamed: 0" in original_feature_names:
            original_feature_names.remove("unnamed: 0")

        # Sort original feature names based on feature_ordering or alphabetically
        if self.feature_ordering is not None:
            sorted_feature_names = sorted(original_feature_names, key=lambda k: self.feature_ordering.index(k))
        else:
            sorted_feature_names = sorted(original_feature_names)

        # Map sorted feature names to their original IDs and display names, if available
        feature_names_id_mapping = [
            {'id': original_feature_names.index(feature_name),
             'feature_name': feature_display_names.get(feature_name, feature_name)}
            for feature_name in sorted_feature_names
        ]

        return feature_names_id_mapping
