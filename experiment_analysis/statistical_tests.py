import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from scipy.stats import mannwhitneyu

import matplotlib
import tikzplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14


def get_groups(df, x_label, y_label, groups_to_compare):
    if groups_to_compare is None:
        group_labels = df[x_label].unique()
    else:
        group_labels = groups_to_compare
    group1 = df[df[x_label] == group_labels[0]][y_label]
    group2 = df[df[x_label] == group_labels[1]][y_label]
    print(f"Group 1: {group_labels[0]}: {len(group1)}")
    print(f"Group 2: {group_labels[1]}: {len(group2)}")
    return group1, group2, group_labels[0], group_labels[1]


def perform_t_test(group1, group2):
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False, alternative='greater')
    return t_stat, p_val


def perform_mann_whitney_u_test(group1, group2):
    U1, p = mannwhitneyu(group1, group2, method="exact", alternative="less")
    return U1, p


def get_max_y_from_seaborn_boxplot(data, y_label):
    # Calculate the IQR
    Q1 = data[y_label].quantile(0.25)
    Q3 = data[y_label].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the maximum y position in the boxplot
    max_y = Q3 + 1.5 * IQR

    return max_y


def plot_box_with_significance_bars(df,
                                    x_label,
                                    y_label,
                                    title,
                                    ttest=False,
                                    ax=None,
                                    y_label_name=None,
                                    groups_to_compare=None):
    """
    Plot boxplot with significance bars based on the result of a statistical test.
    tttest: If True, perform t-test. Otherwise, perform Mann-Whitney U test.
    """
    group1, group2, group_1_name, group_2_name = get_groups(df, x_label, y_label, groups_to_compare)

    # Perform statistical tests for each pair of groups
    if ttest:
        stat12, p_value12 = perform_t_test(group1, group2)
        print(f"Performing t-test for {y_label}, p-values={p_value12}, p-stat={stat12}")
        # title = "T-Test: " + title
    else:
        stat12, p_value12 = perform_mann_whitney_u_test(group1, group2)
        print(f"Performing U test for {y_label}, p-values={p_value12}, u-stat={stat12} ")
        # title = "U-Test: " + title

    """if df2 is None:
        # Plot boxplot only for group2
        df = df[df[x_label] == group_2_name]"""

    # Set the categories for the boxplot
    categories = [group_1_name, group_2_name]
    # Convert categories to string if not already
    categories = [str(cat) for cat in categories]

    colors = {'static': 'darkgrey', 'interactive': 'skyblue', 'chat': 'darkgrey', 'active_chat': 'darkgrey'}

    # Begin plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    sns.boxplot(x=x_label, y=y_label, data=df, order=categories, width=0.3, palette=colors, ax=ax,
                showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(y_label_name)
    ax.set_xlabel("")

    # Get the positions of the boxplots
    box_plot_positions = [i for i in range(len(categories))]

    # Determine the highest point on the y-axis for plotting the significance bar
    y_max = get_max_y_from_seaborn_boxplot(df, y_label) + 1
    print(f"y_max: {y_max}")
    h = 0.5  # Increase the height of the significance bar for more space between bars
    col = 'k'  # Color of the significance bar and text
    lw = 1.5  # Increase the linewidth for bigger beginning and end bars

    # Create a list of tuples containing the statistics and p-values
    stats_and_p_values = [(stat12, p_value12)]

    for i, (stat, p_value) in enumerate(stats_and_p_values):
        if p_value is not None and p_value < 0.05:
            y = y_max + i * h  # Adjust the height based on the loop index
            ax.plot([box_plot_positions[0], box_plot_positions[1]], [y, y], color=col, lw=lw)
            ax.plot([box_plot_positions[0], box_plot_positions[0]], [y - h / 2, y + h / 2], color=col, lw=lw)
            ax.plot([box_plot_positions[1], box_plot_positions[1]], [y - h / 2, y + h / 2], color=col, lw=lw)
            sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
            ax.text((box_plot_positions[0] + box_plot_positions[1]) / 2, y + h / 5, sig_text, ha='center', va='bottom',
                    color=col)

    title = title + "_" + group_1_name + " vs " + group_2_name
    # Set the x-axis tick labels if necessary
    plt.xticks(range(0, len(categories)), categories)
    plt.tight_layout()
    csv_path = "analysis_plots/" + f"{title}.csv"
    df_to_export = df[[x_label, y_label]].copy()
    df_to_export.to_csv(csv_path, index=False)


def is_t_test_applicable(df, x_label, y_label, groups_to_compare=None):
    """
    Check if the t-test assumptions are met
    """
    group1, group2, _, _ = get_groups(df, x_label, y_label, groups_to_compare)
    # Check for normality
    for group in groups_to_compare:
        stat, p_value = stats.shapiro(df[df[x_label] == group][y_label])
        if p_value < 0.05:
            # print(f"Shapiro-Wilk test for {group} failed: P-value={p_value}")
            return False
    # Check for homogeneity of variance
    levene_test = stats.levene(group1, group2)
    if levene_test.pvalue < 0.05:
        # print(f"Levene test failed: P-value={levene_test.pvalue}")
        return False

    return True


def print_correlation_ranking(df, target_var, group=None, keep_cols=None):
    # Make correlation df for user_df with each column against score_improvement
    if group is not None:
        correlation_df = df[df["study_condition"] == group]
    else:
        correlation_df = df
    if keep_cols is None:
        keep_cols = df.columns
        # Remove id column and study_condition
        keep_cols = keep_cols[~keep_cols.isin(["id", "study_condition"])]
    correlation_df = correlation_df[keep_cols]

    correlation_df = correlation_df.select_dtypes(include=[np.number])
    correlation_df = correlation_df.corr()
    correlation_df = correlation_df[target_var].reset_index()
    correlation_df.columns = ["column", "correlation"]
    correlation_df = correlation_df.sort_values("correlation", ascending=False)
    if group is not None:
        print(f"Correlation ranking for {group}", target_var)
    print(correlation_df)
