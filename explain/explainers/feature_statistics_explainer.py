import base64
import io

import pandas as pd

from data.response_templates.feature_statistics_template import feature_statistics_template
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureStatisticsExplainer:
    def __init__(self,
                 data: pd.DataFrame,
                 y_labels: pd.Series,
                 numerical_features: list,
                 feature_names: list,
                 categorical_mapping,
                 rounding_precision: int = 2,
                 feature_units: dict = None
                 ):
        self.data = data
        self.y_labels = y_labels
        self.numerical_features = numerical_features
        self.feature_names = feature_names
        self.rounding_precision = rounding_precision
        self.categorical_mapping = categorical_mapping
        self.feature_units = feature_units

    """def get_categorical_statistics(self, feature_name, as_string=True): # TODO: Check before usage. Old code.
        #Returns a string with the frequencies of values of a categorical feature.
    
        feature_value_frequencies = self.data[feature_name].value_counts()
        # Map feature indeces to feature names
        feature_id = self.data.columns.get_loc(feature_name)
        feature_value_frequencies.index = self.categorical_mapping[feature_id]
        # Sort by frequency
        feature_value_frequencies.sort_values(ascending=False, inplace=True)

        if not as_string:
            return feature_value_frequencies

        result_text = ""
        for i, (value, frequency) in enumerate(feature_value_frequencies.items()):
            result_text += f"The value <b>{value}</b> occurs <b>{frequency}</b> times.<br>"
        return result_text"""

    def get_numerical_statistics(self, feature_name, template_manager, as_string=True, as_plot=False):
        """
        Returns a string with the mean, standard deviation, minimum and maximum values of a numerical feature.
        If as_string is False, returns a tuple with the mean, standard deviation, minimum and maximum values.
        If as_plot is True, returns a plot of the feature distribution.
        """
        if as_plot:
            # return self.explain_numerical_statistics_as_plot(self.data[feature_name], feature_name)
            return self.plot_binary_class_kde(self.data[feature_name], self.y_labels, feature_name)

        mean = round(self.data[feature_name].mean(), 2)
        std = round(self.data[feature_name].std(), 2)
        min_v = round(self.data[feature_name].min(), 2)
        max_v = round(self.data[feature_name].max(), 2)
        # make float values to strings where the . is replaced by a ,
        mean = str(mean).replace(".", ",")
        std = str(std).replace(".", ",")
        min_v = str(min_v).replace(".", ",")
        max_v = str(max_v).replace(".", ",")
        if not as_string:
            return mean, std, min_v, max_v
        return feature_statistics_template(feature_name, mean, std, min_v, max_v, self.feature_units, template_manager)

    def get_categorical_frequencies_fig(self, value_counts, feature_name, as_html=True):
        """
        Creates a bar plot for the frequencies of values for a categorical feature and returns the figure.

        Parameters:
        - value_counts: pandas Series containing the value counts to plot.
        - feature_name: String, the name of the categorical feature to plot.

        Returns:
        - fig: A matplotlib Figure object of the feature value frequencies bar chart.
        """
        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create a bar plot on the axis
        value_counts.plot(kind='bar', ax=ax)
        ax.set_title(f'Frequency of {feature_name} Categories')
        ax.set_xlabel('Category')
        ax.set_ylabel('Frequency')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        feature_id = self.data.columns.get_loc(feature_name)
        categorical_mapping_for_feature_list = self.categorical_mapping.get(feature_id, None)
        categorical_mapping_for_feature_dict = {i: v for i, v in enumerate(categorical_mapping_for_feature_list)}
        ax = plt.gca()  # get current axis

        # Update the x-axis ticks
        ax.set_xticks(list(categorical_mapping_for_feature_dict.keys()))
        ax.set_xticklabels(list(categorical_mapping_for_feature_dict.values()))

        plt.tight_layout()

        if not as_html:
            return fig

        # turn to html
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        html_string = f'<img src="data:image/png;base64,{image_base64}" alt="Your Plot">'

        return html_string

    def get_categorical_statistics(self, feature_name, as_string=True, as_plot=False):
        """
        Returns statistics or a plot for the frequencies of values of a categorical feature.

        Parameters:
        - feature_name: String, the name of the categorical feature.
        - as_string: Boolean, if True, returns a string; otherwise, returns value counts.
        - as_plot: Boolean, if True, returns a plot based on the calculated frequencies.

        Returns:
        - A string with the frequencies, a pandas Series of frequencies, or a matplotlib Figure object.
        """
        feature_value_frequencies = self.data[feature_name].value_counts()
        # Map feature indices to feature names if needed
        if feature_name in self.categorical_mapping:
            feature_id = self.data.columns.get_loc(feature_name)
            feature_value_frequencies.index = self.categorical_mapping[feature_id]
        feature_value_frequencies.sort_values(ascending=False, inplace=True)

        if as_plot:
            return self.get_categorical_frequencies_fig(feature_value_frequencies, feature_name)

        if not as_string:
            return feature_value_frequencies

        result_text = ""
        for value, frequency in feature_value_frequencies.items():
            result_text += f"The value <b>{value}</b> occurs <b>{frequency}</b> times.<br>"
        return result_text

    def get_single_feature_statistic(self, feature_name, template_manager, as_string=True):
        # Check if feature is numerical or categorical
        if feature_name in self.numerical_features:
            return self.get_numerical_statistics(feature_name, template_manager, as_string=as_string)
        else:
            return self.get_categorical_statistics(feature_name, as_plot=True)

    def get_all_feature_statistics(self, template_manager, as_string=True):
        feature_stats = {}
        for feature_name in self.feature_names:
            # Make dict of feature names to feature statistics (min, max, mean)
            if feature_name in self.numerical_features:
                if as_string:
                    feature_stats[feature_name] = self.get_numerical_statistics(feature_name, template_manager,
                                                                                as_string=as_string)
                else:
                    mean, _, min_v, max_v = self.get_numerical_statistics(feature_name, template_manager,
                                                                          as_string=as_string)
                    feature_stats[feature_name] = {"mean": mean, "min": min_v, "max": max_v}
            else:
                feature_stats[feature_name] = self.get_categorical_statistics(feature_name, as_plot=True)
        return feature_stats

    def explain_numerical_statistics_as_plot(self, feature_data, feature_name):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import io
        import base64

        # Create the KDE plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(feature_data, ax=ax, color='blue', shade=True)

        # Setting labels and title
        ax.set_title(f"Data Density Plot of {feature_name}", fontsize=20)
        ax.set_xlabel(feature_name, fontsize=18)
        ax.set_ylabel("Density", fontsize=18)

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Clear the current plot to free memory
        plt.close()

        html_string = f'<img src="data:image/png;base64,{image_base64}" alt="KDE Plot">' \
                      f'<span>A smoothed representation showing where most of the data points are concentrated.</span>'

        return html_string

    def plot_binary_class_kde(self, data, labels, feature_name,
                              class_names=['unlikely to have diabetes', 'likely to have diabetes']):
        """
        Plots KDEs for a binary class feature.

        Parameters:
        - data (pd.Series or np.array): The feature data.
        - labels (pd.Series or np.array): Binary labels corresponding to the data.
        - feature_name (str): Name of the feature for labeling the x-axis.
        - class_names (list): Names of the two classes. Default is ['Class 0', 'Class 1'].
        """

        # Split the data based on the labels
        class_0_data = data[labels == 0]
        class_1_data = data[labels == 1]

        # Plotting the KDEs
        plt.figure(figsize=(10, 6))
        sns.kdeplot(class_0_data, shade=True, label=class_names[0], color='blue')
        sns.kdeplot(class_1_data, shade=True, label=class_names[1], color='red')

        # Setting title and labels
        plt.title("Data Density Plot by Class")
        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def get_feature_ranges(self):
        """
        Return the feature ranges for all features in the dataset, either min, max for numerical features or
        the unique values for categorical features.
        """
        feature_ranges = {}
        for feature_name in self.feature_names:
            if feature_name in self.numerical_features:
                feature_ranges[feature_name] = (self.data[feature_name].min(), self.data[feature_name].max())
            else:
                feature_ranges[feature_name] = self.data[feature_name].unique()
