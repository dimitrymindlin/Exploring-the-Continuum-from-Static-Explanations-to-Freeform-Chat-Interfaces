import warnings
from typing import List

import gin
import pandas as pd
from tqdm import tqdm
from explain.explanation import Explanation
import dalex as dx
from scipy import interpolate
import numpy as np

"""def plot_cp(names, values):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the y-axis labels
    ax.set_yticklabels(names)

    # Set the y-axis ticks
    ax.set_yticks(range(len(names)))

    # Set the x-axis range
    ax.set_xlim(min(values), max(values))

    plt.subplots_adjust(left=0.5)

    # Plot the bars
    for i, val in enumerate(values):
        color = 'green' if val > 0 else 'red'
        ax.barh(i, val, color=color)
    plt.show()"""


def find_x_for_y_plotly(fig, y_target=0.5):
    # Assuming the first trace contains the relevant data
    trace = fig.data[0]
    x_data = np.array(trace.x)
    y_data = np.array(trace.y)

    # Interpolating
    f = interpolate.interp1d(y_data, x_data, bounds_error=False, fill_value='extrapolate')
    x_at_y_target = f(y_target)

    round_x_at_y_target = np.round(x_at_y_target, 2)
    return round_x_at_y_target


def find_categories_crossing_threshold_scatter(fig, threshold, current_feature_value):
    crossing_categories = []

    def get_categories_and_values_dict(trace):
        x_data = np.array(trace.x)
        y_data = np.array(trace.y)
        indices = np.where(x_data == current_feature_value)[0]

        # Check if indices are found; if not, return None for current_proba
        if len(indices) == 0:
            return None, {}

        current_proba = y_data[indices][0]
        # Since we're interested in comparing to other categories, remove the current feature value
        x_data = np.delete(x_data, indices)
        y_data = np.delete(y_data, indices)

        # Make a dict from x_data to y_data
        category_proba_dict = dict(zip(x_data, y_data))
        return current_proba, category_proba_dict

    for trace in fig.data:
        current_probability, category_proba_dict = get_categories_and_values_dict(trace)

        # Check if current_probability is None (current_feature_value was not found)
        if current_probability is None:
            continue

        for category, proba in category_proba_dict.items():
            if proba < threshold and current_probability > threshold:
                crossing_categories.append(category)
            elif proba > threshold and current_probability < threshold:
                crossing_categories.append(category)

    return crossing_categories


@gin.configurable
class CeterisParibus(Explanation):
    """This class generates CeterisParibus explanations for tabular data."""

    def __init__(self,
                 model,
                 background_data: pd.DataFrame,
                 ys: pd.DataFrame,
                 class_names: dict,
                 cache_location: str = "./cache/ceterisparibus-tabular.pkl",
                 feature_names: list = None,
                 categorical_mapping: dict = None,
                 ordinal_features: List[str] = None):
        """

        Args:
            model: The model to explain.
            data: the background dataset provided at pandas df
            value of the categorical features. Every feature that is not in
            this map will be considered as ordinal or continuous, and thus discretized.
            class_names: The names of the classes.
            cache_location: The location to save the cache.
            feature_names: The names of the features.
            categorical_mapping: The mapping of the categorical features to their values.
        """
        super().__init__(cache_location, class_names)
        self.background_data = background_data
        self.model = model
        self.feature_names = feature_names
        self.ys = ys
        self.explainer = dx.Explainer(self.model, self.background_data, y=self.ys)
        self.categorical_mapping = categorical_mapping

    def run_explanation(self,
                        current_data: pd.DataFrame):
        """Generate ceteris paribus explanations.

        Arguments:
            current_data: The data to generate explanations for in pandas df.
            desired_class: The desired class of the cfes. If None, will use the default provided
                           at initialization.
        Returns:
            explanations: The generated cf explanations.
        """
        cps = {}
        for d in tqdm(list(current_data.index)):
            instance = current_data.loc[[d]]
            rf_profile = self.explainer.predict_profile(instance)
            cps[d] = rf_profile
        return cps

    def get_explanation(self, data_df, feature_name=None, as_plot=True):
        id = data_df.index[0]
        cp_data = self.get_explanations([id], self.background_data, save_to_cache=True)
        feature_id = self.feature_names.index(feature_name)
        if not as_plot:
            return cp_data
        else:
            if feature_id in self.categorical_mapping.keys():
                variables_type = 'categorical'
            else:
                variables_type = 'numerical'
            fig = cp_data[id].plot(variables=[feature_name], show=False, variable_type=variables_type)
            # Update the y-axis tick labels
            if feature_id in self.categorical_mapping.keys():
                categorical_mapping_for_feature = self.categorical_mapping[feature_id]
                # Convert the list to a dictionary
                categorical_mapping_for_feature_dict = {i: val for i, val in enumerate(categorical_mapping_for_feature)}
                fig.update_yaxes(tickvals=list(categorical_mapping_for_feature_dict.keys()),
                                 ticktext=list(categorical_mapping_for_feature_dict.values()))
                # Sort the y-axis tick labels alphabetically
                fig.update_yaxes(categoryorder='array', categoryarray=categorical_mapping_for_feature)
            return fig

    def get_feature_values_flipping_prediction(self, data_df, feature_name=None):
        def check_x_value_in_range(x_value):
            # check if x_value is in the range of the feature
            feature_max = self.background_data[feature_name].max()
            feature_min = self.background_data[feature_name].min()
            if x_value > feature_max or x_value < feature_min:
                x_value = None
            return x_value

        id = data_df.index[0]
        current_feature_value = data_df[feature_name].values[0]
        cp_data = self.get_explanations([id], self.background_data, save_to_cache=True)
        fig = cp_data[id].plot(variables=[feature_name], show=False)
        # For numerical feature, find the x value for y = 0.5
        feature_id = self.feature_names.index(feature_name)
        if feature_id in self.categorical_mapping.keys():
            x_value = find_categories_crossing_threshold_scatter(fig, 0.5, current_feature_value)
        else:
            x_value = find_x_for_y_plotly(fig, 0.5)

        possible_x_values = []
        # Check if x_value is a list
        if isinstance(x_value, list):
            for x in x_value:
                x_value = check_x_value_in_range(x)
                if x_value is not None:
                    possible_x_values.append(x_value)
        else:
            x_value = check_x_value_in_range(x_value)
            if x_value is not None:
                possible_x_values.append(x_value)

        return possible_x_values
