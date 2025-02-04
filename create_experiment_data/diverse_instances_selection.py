import os
import time
from typing import List
import pickle as pkl
import gin
import pandas as pd
import numpy as np


def load_cache(cache_location: str):
    """Loads the cache."""
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            cache = pkl.load(file)
    else:
        cache = []
    return cache


@gin.configurable
class DiverseInstances:
    """This class finds DiverseInstances by using LIMEs submodular pick."""

    def __init__(self,
                 cache_location: str = "./cache/diverse-instances.pkl",
                 dataset_name: str = "german",
                 instance_amount: int = 5,
                 lime_explainer=None):
        """

        Args:
            cache_location: location to save the cache
            lime_explainer: lime explainer to use for finding diverse instances (from MegaExplainer)
        """
        self.diverse_instances = load_cache(cache_location)
        self.cache_location = cache_location
        self.lime_explainer = lime_explainer
        self.instance_amount = instance_amount
        self.dataset_name = dataset_name

    def filter_instances_by_class(self, data, model, diverse_instances_pandas_indices,
                                  instance_amount, filter_by_additional_feature=False):
        # Step 1: Predict Classes
        predicted_classes = model.predict(data.loc[diverse_instances_pandas_indices])

        # Step 2: Separate Instances by Predicted Class
        class_0_indices, class_1_indices = [], []
        for i, pred_class in zip(diverse_instances_pandas_indices, predicted_classes):
            if pred_class == 0:
                class_0_indices.append(i)
            else:
                class_1_indices.append(i)

        # Helper function to filter instances based on alternating marital status
        def filter_by_marital_status(indices):
            filtered = []
            last_marital_status = -1
            for i in indices:
                current_marital_status = data.loc[i, "maritalstatus"]
                if current_marital_status != last_marital_status:
                    filtered.append(i)
                    last_marital_status = current_marital_status
            return filtered

        # Step 3: Filter each class list by marital status
        if filter_by_additional_feature:
            class_0_indices = filter_by_marital_status(class_0_indices)
            class_1_indices = filter_by_marital_status(class_1_indices)

        # Step 4: Balance the classes
        min_length = min(len(class_0_indices), len(class_1_indices))
        balanced_class_0 = np.random.choice(class_0_indices, min_length, replace=False)
        balanced_class_1 = np.random.choice(class_1_indices, min_length, replace=False)

        # Combine the lists
        combined_instances = np.concatenate((balanced_class_0, balanced_class_1))

        # Step 5: Shuffle and select the final set of instances
        np.random.shuffle(combined_instances)
        final_instances = combined_instances[:instance_amount].tolist()

        return final_instances

    def get_instance_ids_to_show(self,
                                 data: pd.DataFrame,
                                 model,
                                 y_values: List[int],
                                 save_to_cache=True,
                                 submodular_pick=False) -> List[int]:
        """
        Returns diverse instances for the given data set.
        Args:
            data: pd.Dataframe the data instances to use to find diverse instances
            instance_count: number of diverse instances to return
            save_to_cache: whether to save the diverse instances to the cache
        Returns: List of diverse instance ids.

        """
        if len(self.diverse_instances) > 0:
            return self.diverse_instances

        counter = 0
        while len(self.diverse_instances) < self.instance_amount:

            # Generate diverse instances
            if submodular_pick:
                print(f"Using submodular pick to find {self.instance_amount} diverse instances.")
                diverse_instances = self.lime_explainer.get_diverse_instance_ids(data.values, self.instance_amount)
                # Get pandas index for the diverse instances
                diverse_instances_pandas_indices = [data.index[i] for i in diverse_instances]
            else:
                # Get random instances
                dynamic_seed = int(time.time()) % 10000
                # Get 10 times more instances to filter and ensure diversity
                diverse_instances_pandas_indices = data.sample(self.instance_amount * 10,
                                                               random_state=dynamic_seed).index.tolist()

                # If adult dataset, filter by marital status and class
                if self.dataset_name == "adult":
                    diverse_instances_pandas_indices = self.filter_instances_by_class(data,
                                                                                      model,
                                                                                      diverse_instances_pandas_indices,
                                                                                      self.instance_amount,
                                                                                      filter_by_additional_feature=True)
                elif self.dataset_name == "german":
                    diverse_instances_pandas_indices = self.filter_instances_by_class(data,
                                                                                      model,
                                                                                      diverse_instances_pandas_indices,
                                                                                      self.instance_amount,
                                                                                      filter_by_additional_feature=False)

            for i in diverse_instances_pandas_indices:
                if i not in self.diverse_instances:
                    self.diverse_instances.append(i)

            counter += 1
            if counter > 20:
                print(f"Could not find enough diverse instances, only found {len(self.diverse_instances)}.")
                break

        # TODO: This is hacky and only for Diabetes dataset. Move to data preprocessing
        """# remove instance with id 123
        diverse_instances_pandas_indices.remove(123)"""

        if save_to_cache:
            with open(self.cache_location, 'wb') as file:
                pkl.dump(diverse_instances_pandas_indices, file)
        return diverse_instances_pandas_indices
