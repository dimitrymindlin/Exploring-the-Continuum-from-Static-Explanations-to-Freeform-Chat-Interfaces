import numpy as np
from typing import Dict
import io


def textual_fi_with_values(sig_coefs, num_features_to_show=None, filtering_text=None, template_manager=None):
    """Formats sorted list of (feature name, feature importance) tuples into a string.

    Arguments:
        sig_coefs: Sorted list of tuples (feature name, feature importance) with textual feature names.
        num_features_to_show: Number of features to show in the output. If None, all features are shown.
        filtering_text: Text to control output formatting (e.g., "least 3", "top 3", "only_positive").
        template_manager: Object to access feature display names.
    Returns:
        String with the formatted feature importances.
    """
    output_text = "<ol>"

    if "least 3" in filtering_text:
        # Handle summarizing least important features separately
        sig_coefs = sig_coefs[::-1]  # Reverse to start with the worst.

    describing_features = 0  # Initialize counter for described features
    for i, (feature_name, feature_importance) in enumerate(sig_coefs):
        if describing_features == 3 or \
                num_features_to_show and describing_features >= num_features_to_show or \
                "only_positive" in filtering_text and feature_importance <= 0:
            break

        feature_display_name = template_manager.get_feature_display_name_by_name(feature_name)
        if i + 1 == 2:
            position_prefix = "second"
        elif i + 1 == 3:
            position_prefix = "third"
        else:
            position_prefix = ""

        if "least" in filtering_text:
            position = "least" if i == 0 else f"{position_prefix} least"
        else:
            position = "most" if i == 0 else f"{position_prefix} most"
        increase_decrease = "increases" if feature_importance > 0 else "decreases"
        # output_text += f"<li><b>{feature_display_name}</b> is the <b>{position}</b> important attribute which {increase_decrease} the model's output.</li>"
        output_text += f"<li><b>{feature_display_name}</b> is the <b>{position}</b> important attribute.</li>"
        describing_features += 1
    output_text += "</ol>"
    return output_text


def textual_fi_relational(sig_coefs: Dict[str, float],
                          num_features_to_show: int = None,
                          print_unimportant_features: bool = False,
                          filtering_text: str = None):
    """Formats dict of label -> feature name -> feature_importance dicts to string.

    Arguments:
        sig_coefs: Dict of label -> feature name -> feature_importance dicts with textual feature names.
    Returns:

    """

    def relational_percentage_to_comparative_language(percentage_number):
        comparative_string = ""
        if percentage_number > 95:
            comparative_string = "almost as important as"
        elif percentage_number > 80:
            comparative_string = "similarly important as"
        elif percentage_number > 60:
            comparative_string = "three fourth as important as"
        elif percentage_number > 40:
            comparative_string = "half as important as"
        elif percentage_number > 20:
            comparative_string = "one fourth as important as"
        elif percentage_number > 5:
            comparative_string = "almost unimportant compared to"
        else:
            comparative_string = "unimportant compared to"
        return comparative_string

    output_text = "<ol>"

    for i, (current_feature_value, feature_importance) in enumerate(sig_coefs):
        if filtering_text == "top 3":
            if i == 3:
                break
        elif filtering_text == "least 3":
            if i < len(sig_coefs) - 3:
                continue
        if i == 0:
            position = "most"
        else:
            position = f"{i + 1}."
        increase_decrease = "increases" if feature_importance > 0 else "decreases"

        ### Get importance comparison strings
        # Most important can be printed as it is
        # TODO: Include this? is the <b>{position}</b> important attribute and it
        new_text = (f"<b>{current_feature_value}</b> "
                    f" <em>{increase_decrease}</em> the likelihood of the current prediction.")
        previous_feature_importance = feature_importance
        # For the rest, we need to check the relation to the previous feature
        if i > 0:
            relation_to_previous_in_percentage = np.abs(
                round((feature_importance / previous_feature_importance) * 100, 2))
            relation_to_top_in_percentage = np.abs(round((feature_importance / sig_coefs[0][1]) * 100, 2))
            if not print_unimportant_features:
                if relation_to_top_in_percentage < 5:
                    break
            comparitive_string_to_top_feature = relational_percentage_to_comparative_language(
                relation_to_top_in_percentage)
            new_text += f" It is {comparitive_string_to_top_feature} the top feature"
            if i > 1:
                comparative_string_to_previous = relational_percentage_to_comparative_language(
                    relation_to_previous_in_percentage)
                new_text += f" and {comparative_string_to_previous} the previous feature"
            new_text += "."

        if new_text != "":
            output_text += "<li>" + new_text + "</li>"
        if num_features_to_show:
            if i == num_features_to_show:
                break
    output_text += "</ol>"
    return output_text


def visual_feature_importance_list(sig_coefs):
    import matplotlib.pyplot as plt

    def plot_values(names, values):
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

        # Save the plot to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Clear the current plot
        plt.clf()

        # Close the figure to release memory
        plt.close(fig)

        # Send the buffer as a file attachment
        return buffer

    names = [sig_coefs[i][0] for i in range(len(sig_coefs))]
    names.reverse()
    values = [sig_coefs[i][1] for i in range(len(sig_coefs))]
    values.reverse()
    plot_values(names, values)
