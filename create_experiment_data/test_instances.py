import os
from typing import List, Dict, Any
import pickle as pkl
import gin
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_cache(cache_location: str):
    """Loads the cache."""
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            cache = pkl.load(file)
    else:
        cache = []
    return cache


@gin.configurable
class TestInstances:
    """This class creates test instances for the diverse instances"""

    def __init__(self,
                 data,
                 model,
                 mega_explainer,
                 experiment_helper,
                 diverse_instance_ids,
                 actionable_features,
                 max_features_to_vary=2,
                 cache_location: str = "./cache/test-instances.pkl",
                 instance_amount: int = 10):
        self.data = data
        self.diverse_instance_ids = diverse_instance_ids
        self.cache_location = cache_location
        self.model = model
        self.mega_explainer = mega_explainer
        self.experiment_helper = experiment_helper
        self.actionable_features = actionable_features
        self.test_instances = load_cache(cache_location)
        self.max_features_to_vary = max_features_to_vary
        self.instance_amount = instance_amount

    def get_random_instance(self):
        return self.data.sample(1)

    def get_test_instances(self,
                           save_to_cache: bool = True,
                           close_instances: bool = True) -> Dict[str, Any]:
        """
        Generates instances for testing based on the provided dataset.

        Parameters:
            instance_count (int): Number of diverse instances to generate.
            save_to_cache (bool): Whether to save the generated instances to cache.
            close_instances (bool): Flag to determine whether to generate close (similar) or random instances.

        Returns:
            Dict[str, Any]: A dictionary with diverse test instances.
        """
        if self.test_instances:
            return self.test_instances

        test_instances = self._generate_instances(self.instance_amount, close_instances)
        self._save_instances_to_cache(test_instances, save_to_cache)
        return test_instances

    def _generate_instances(self,
                            instance_count: int,
                            close_instances: bool) -> Dict[str, Any]:
        """
        Generates diverse instances for testing based on the provided dataset.
        param: instance_count (int): Number of diverse instances to generate.
        param: close_instances (bool): Flag to determine whether to generate close (similar) or random instances.
        """
        test_instances = {}
        for instance_id in self.diverse_instance_ids:
            original_instance = self._get_instance_dataframe(instance_id)
            if not close_instances:
                test_instances[instance_id] = self._generate_random_instances()
            else:
                test_instances[instance_id] = self._generate_close_instances(original_instance, instance_count)
        return test_instances

    def _get_instance_dataframe(self, instance_id: str) -> pd.DataFrame:
        return self.data.loc[instance_id].to_frame().transpose()

    def _generate_random_instances(self) -> Dict[str, Any]:
        test_instance = self.get_random_instance()
        # add label of the instance
        test_instance['label'] = self.model.predict(test_instance)
        return {
            "most_complex_instance": test_instance,
            "least_complex_instance": test_instance,
            "easy_counterfactual_instance": test_instance,
            "hard_counterfactual_instance": test_instance
        }

    def _generate_close_instances(self,
                                  original_instance: pd.DataFrame,
                                  instance_count: int) -> Dict[str, Any]:
        original_class_prediction, feature_importances = self._predict_and_get_importances(original_instance)
        similar_instances = self._get_similar_instances(original_instance, instance_count, original_class_prediction)
        complex_instances = self._sort_and_select_instances(original_instance, similar_instances, feature_importances)
        counterfactuals = self._get_and_sort_counterfactuals(original_instance,
                                                             complex_instances['least_complex_instance'],
                                                             feature_importances)
        return {**complex_instances, **counterfactuals}

    def _predict_and_get_importances(self, instance: pd.DataFrame) -> tuple:
        prediction = np.argmax(self.model.predict_proba(instance)[0])
        importances = self.mega_explainer.get_feature_importances(instance)[0][prediction]
        return prediction, importances

    def _get_similar_instances(self, original_instance: pd.DataFrame,
                               instance_count: int,
                               original_class_prediction: int,
                               only_correct_model_predictions: bool = True) -> pd.DataFrame:
        max_attempts = 10
        attempt_count = 0
        while attempt_count < max_attempts:
            attempt_count += 1
            similar_instances_list = []
            for _ in range(instance_count):
                num_changed_features = 0
                similar_instance = None
                while num_changed_features <= 1:  # This could be replaced with a parameter
                    similar_instance, num_changed_features = self.experiment_helper.get_similar_instance(
                        original_instance if num_changed_features == 0 else similar_instance,
                        self.model,
                        max_features_to_vary=self.max_features_to_vary
                    )
                if not isinstance(similar_instance, pd.DataFrame) and similar_instance is not None:
                    similar_instance = pd.DataFrame([similar_instance], columns=original_instance.columns)
                similar_instances_list.append(similar_instance)

            # Concatenate into a single DataFrame.
            similar_instances = pd.concat(similar_instances_list).reset_index(drop=True)

            # Apply model predictions in a batch for efficiency
            predictions = np.argmax(self.model.predict_proba(similar_instances), axis=1)

            # Filter based on prediction criteria.
            if only_correct_model_predictions:
                # check how many instances have the same prediction as the original instance
                same_prediction_count = sum(predictions == original_class_prediction)
                if same_prediction_count == 0:
                    print(f"Attempt {attempt_count} failed: No similar instances with correct prediction.")
                    continue  # Start over and count this as a failed attempt
                similar_instances = similar_instances[predictions == original_class_prediction]

            if len(similar_instances) > 0:
                return similar_instances

        # If maximum attempts are reached and no valid instances found, raise an error
        raise ValueError(f"Failed to find similar instances with correct predictions after {max_attempts} attempts.")

    def _sort_and_select_instances(self,
                                   original_instance: pd.DataFrame,
                                   instances: pd.DataFrame,
                                   feature_importances: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        Sorts instances based on their complexity using feature importances and model predictions,
        then selects the most and least complex instances.

        Parameters:
            original_instance (pd.DataFrame): The original instance for complexity comparison.
            instances (pd.DataFrame): A DataFrame of instances to sort.
            feature_importances (pd.Series): Feature importances from the original instance.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the most and least complex instances.
        """
        original_logits = self.model.predict_proba(original_instance)[0]
        complexities = []

        # Calculate complexities for each instance
        for _, row in instances.iterrows():
            instance = pd.DataFrame(row).transpose()
            instance = instance.astype(original_instance.dtypes.to_dict())  # Match data types
            instance_logits = self.model.predict_proba(instance)[0]
            complexity = self._calculate_prediction_task_complexity(original_instance, instance, feature_importances,
                                                                    original_logits, instance_logits)
            complexities.append(complexity)

        instances['complexity'] = complexities
        sorted_instances = instances.sort_values(by='complexity', ascending=False)

        # Remove the 'complexity' column since not needed further
        sorted_instances.drop(columns='complexity', inplace=True)

        # Extract the most and least complex instances based on calculated complexities
        most_complex_instance = sorted_instances.head(1)
        least_complex_instance = sorted_instances.tail(1)

        return {
            "most_complex_instance": most_complex_instance,
            "least_complex_instance": least_complex_instance
        }

    def _get_and_sort_counterfactuals(self,
                                      original_instance: pd.DataFrame,
                                      least_complex_instance: pd.DataFrame,
                                      feature_importances: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        Generates and sorts counterfactual instances based on their complexity,
        using the least complex instance and feature importances.

        Parameters:
            original_instance (pd.DataFrame): The original instance for reference.
            least_complex_instance (pd.DataFrame): The identified least complex instance.
            feature_importances (pd.Series): Feature importances from the original instance.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing easy and hard counterfactual instances.
        """
        # Generate counterfactual instances.
        counterfactual_instances = None
        counter = 0
        while counterfactual_instances is None:
            counterfactual_instances = self.experiment_helper.get_counterfactual_instances(least_complex_instance)
            counter += 1
            if counter > 10:
                raise ValueError("Failed to generate counterfactual instances after 10 attempts in test instances"
                                 "for id {}".format(original_instance.index[0]))

        # Calculate original and counterfactual logits for complexity calculation
        original_logits = self.model.predict_proba(original_instance)[0]

        # Initialize a list to hold complexities
        complexities = []

        # Iterate over counterfactual instances to calculate complexities
        for _, row in counterfactual_instances.iterrows():
            cf_instance = pd.DataFrame(row).transpose()
            cf_instance = cf_instance.astype(original_instance.dtypes.to_dict())  # Ensure dtype consistency
            cf_logits = self.model.predict_proba(cf_instance)[0]

            complexity = self._calculate_prediction_task_complexity(original_instance, cf_instance, feature_importances,
                                                                    original_logits, cf_logits)
            complexities.append(complexity)

        counterfactual_instances['complexity'] = complexities

        # Sort counterfactual instances by complexity
        sorted_counterfactuals = counterfactual_instances.sort_values(by='complexity', ascending=False)

        # Remove the 'complexity' since not needed further
        sorted_counterfactuals.drop(columns='complexity', inplace=True)

        # Selecting the easy and hard counterfactual instances
        easy_counterfactual_instance = sorted_counterfactuals.tail(1)
        hard_counterfactual_instance = sorted_counterfactuals.head(1)

        return {
            "easy_counterfactual_instance": easy_counterfactual_instance,
            "hard_counterfactual_instance": hard_counterfactual_instance
        }

    def _save_instances_to_cache(self, instances: Dict[str, Any], save_to_cache: bool) -> None:
        if save_to_cache:
            with open(self.cache_location, 'wb') as file:
                pkl.dump(instances, file)

    def _calculate_prediction_task_complexity(self, original_instance, new_instance, feature_importances,
                                              model_certainty_old, model_certainty_new):
        # Ensure both instances are DataFrames
        if not isinstance(original_instance, pd.DataFrame):
            original_instance = pd.DataFrame(original_instance).transpose()
        if not isinstance(new_instance, pd.DataFrame):
            new_instance = pd.DataFrame(new_instance).transpose()

        # Normalize feature importances
        total_importance = sum([item for sublist in feature_importances.values() for item in sublist])
        normalized_importances = {k: v / total_importance for k, v in feature_importances.items()}

        # Calculate weighted sparsity of changes
        new_instance_values = new_instance.values.astype(float)
        original_instance_values = original_instance.values.astype(float)
        changes = np.abs(new_instance_values - original_instance_values) * [normalized_importances.get(col, 0) for col
                                                                            in original_instance.columns]
        weighted_sparsity = np.sum(changes)

        # Calculate similarity (1 - cosine similarity)
        similarity = cosine_similarity(original_instance.values.reshape(1, -1), new_instance.values.reshape(1, -1))[0][
            0]
        similarity_score = 1 - similarity

        # Integrate model certainty change if necessary
        # certainty_change = np.abs(model_certainty_new - model_certainty_old)

        # Combine measures into a final complexity score
        complexity_score = weighted_sparsity + similarity_score  # + certainty_change

        return complexity_score

    def _sort_instances_by_complexity(self, original_instance: pd.DataFrame, instances: pd.DataFrame,
                                      feature_importances: pd.Series) -> pd.DataFrame:
        """
        Sorts instances based on their prediction task complexity using feature importances and model predictions.

        Parameters:
            original_instance (pd.DataFrame): The original instance as a DataFrame.
            instances (pd.DataFrame): New instances as a DataFrame.
            feature_importances (pd.Series): Feature importances for the original instance.

        Returns:
            pd.DataFrame: Instances sorted by their prediction task complexity.
        """
        # Ensure all instances are in a DataFrame
        if isinstance(instances, list):
            instances = pd.concat(instances)

        original_logits = self.model.predict_proba(original_instance)[0]
        instance_complexities = []

        # Iterate over new instances DataFrame
        for index, row in instances.iterrows():
            new_instance = pd.DataFrame(row).transpose()
            # Ensure new instance has the same data type as the original instance for all columns
            new_instance = new_instance.astype(original_instance.dtypes.to_dict())

            # Attempt prediction and handle potential ValueError by adjusting data types as necessary
            try:
                new_instance_logits = self.model.predict_proba(new_instance)[0]
            except ValueError:
                new_instance = new_instance.astype('float64')
                new_instance_logits = self.model.predict_proba(new_instance)[0]

            # Calculate prediction task complexity for the new instance
            instance_complexity = self._calculate_prediction_task_complexity(original_instance, new_instance,
                                                                             feature_importances, original_logits,
                                                                             new_instance_logits)
            instance_complexities.append(instance_complexity)

        # Add complexities to the DataFrame and sort
        instances['complexity'] = instance_complexities
        sorted_instances = instances.sort_values(by='complexity', ascending=False)

        # Optionally drop the 'complexity' column if not needed for further analysis
        sorted_instances.drop(columns='complexity', inplace=True)

        return sorted_instances
