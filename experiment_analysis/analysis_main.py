import pandas as pd
from matplotlib import pyplot as plt

from load_data import load_data
from plot_overviews import plot_questions_tornado
from process_mining.process_mining import ProcessMining
from statistical_tests import print_correlation_ranking

analysis_steps = ["print_buttons_feedback"]

# Map question ids of interactive from int to str
id_to_question_mapping = {
    23: "top3Features",
    11: "anchor",
    24: "shapAllFeatures",
    27: "least3Features",
    25: "ceterisParibus",
    13: "featureStatistics",
    7: "counterfactualAnyChange",
    0: "followUp",
    1: "whyExplanation",
    100: "notXaiMethod",
    99: "greeting",
    -1: "None"
}


# print_buttons_feedback, plot_question_raking, print_correlations

def get_wort_and_best_users(df_with_score, score_name, group_name):
    # Filter df_with_score by group_name
    df_with_score = df_with_score[df_with_score["study_condition"] == group_name]
    # Calculate thresholds for the top 5% and bottom 5% based on the score
    top_5_percent_threshold = df_with_score[score_name].quantile(0.85)
    bottom_5_percent_threshold = df_with_score[score_name].quantile(0.15)

    # Filter best users = final_score >= top 5% threshold
    best_users = df_with_score[df_with_score[score_name] >= top_5_percent_threshold]

    # Filter worst users = final_score <= bottom 5% threshold
    worst_users = df_with_score[df_with_score[score_name] <= bottom_5_percent_threshold]

    if len(worst_users) > len(best_users):
        # Order by final_score and take the worst users
        worst_users = worst_users.sort_values("final_score", ascending=True)
        worst_users = worst_users.head(len(best_users))
    else:
        # Order by final_score and take the worst users
        best_users = best_users.sort_values("final_score", ascending=False)
        best_users = best_users.head(len(worst_users))
    # return user ids
    return best_users["id"].to_list(), worst_users["id"].to_list()


def make_question_count_df(questions_over_time_df, user_df, chat=True):
    # For each user calculate the frequency of each question type
    user_question_freq = questions_over_time_df.groupby("user_id")["question_id"].value_counts().unstack()
    # Replace NaN with 0
    user_question_freq = user_question_freq.fillna(0)
    # Combine question id 25 and question id 13 to question id 99
    if chat:
        user_question_freq["feature_specific"] = user_question_freq['ceterisParibus'] + user_question_freq[
            "featureStatistics"]
    else:
        user_question_freq["feature_specific"] = user_question_freq[25] + user_question_freq[13]
    # Add final_score to the user_question_freq
    # Combine the rest of the questions to question id 100
    if chat:
        user_question_freq["general"] = user_question_freq['counterfactualAnyChange'] + user_question_freq[
            'top3Features'] + \
                                        user_question_freq['anchor'] + user_question_freq['least3Features'] + \
                                        user_question_freq['shapAllFeatures']
    else:
        user_question_freq["general"] = user_question_freq[7] + user_question_freq[23] + user_question_freq[11] + \
                                        user_question_freq[27] + user_question_freq[24]

    user_question_freq = user_question_freq.merge(user_df[["id", "final_irt_score", "study_condition"]],
                                                  left_on="user_id", right_on="id")
    return user_question_freq


def filter_by_group(df, group_name):
    return df[df['study_condition'] == group_name]


chat_path = "/Users/dimitrymindlin/UniProjects/Dialogue-XAI-APP/experiment_analysis/data_chat_final"
interactive_path = "/Users/dimitrymindlin/UniProjects/Dialogue-XAI-APP/experiment_analysis/data_static_interactive"


def main(groups_to_compare):
    data_static_interactive = load_data(interactive_path)
    data_chat = load_data(chat_path)
    groups_data = []

    # Replace the "user_df" with the "user_df" from the saved data
    static_user_df = pd.read_csv("static_interactive_user_df_with_irt_scores_clean.csv")
    chat_user_df = pd.read_csv("chat_user_df_with_irt_scores_clean.csv")
    data_static_interactive["user_df"] = static_user_df
    data_chat["user_df"] = chat_user_df

    active = filter_by_group(data_chat['user_df'], 'active_chat')
    chat = filter_by_group(data_chat['user_df'], 'chat')

    interactive = \
        data_static_interactive["user_df"][data_static_interactive["user_df"]["study_condition"] == "interactive"]
    static = data_static_interactive["user_df"][data_static_interactive["user_df"]["study_condition"] == "static"]

    # Print the number of users in each group
    print("Number of users in each group:")
    print(f"Active Chat: {len(active)}")
    print(f"Chat: {len(chat)}")
    print(f"Interactive: {len(interactive)}")
    print(f"Static: {len(static)}")

    for idx, group in enumerate(groups_to_compare):
        if group == "chat":
            groups_data.append(chat)
        elif group == "active_chat":
            groups_data.append(active)
        elif group == "static":
            groups_data.append(static)
        elif group == "interactive":
            groups_data.append(interactive)

    ### Plot Analysis

    # Plot plotly pie charts
    if "plot_failure_pie_charts" in analysis_steps:
        # Assuming data_chat is your original DataFrame
        data = []

        for user_group in ["active_chat", "chat"]:
            category_frequencies_most = data_chat["user_df"][data_chat["user_df"]["study_condition"] == user_group][
                "understanding_question_most_important"].value_counts()
            category_frequencies_least = data_chat["user_df"][data_chat["user_df"]["study_condition"] == user_group][
                "understanding_question_least_important"].value_counts()

            for category, value in category_frequencies_most.items():
                data.append([user_group, 'Most Important', category, value])
            for category, value in category_frequencies_least.items():
                data.append([user_group, 'Least Important', category, value])

        df = pd.DataFrame(data, columns=['Group', 'Condition', 'Category', 'Value'])

        # Print the DataFrame
        print(df)

        # Bar Chart
        def plot_bar_chart(df):
            fig, ax = plt.subplots(figsize=(12, 6))

            for (condition, group), group_data in df.groupby(['Condition', 'Group']):
                group_data.plot(kind='bar', x='Category', y='Value', ax=ax, label=f'{group} - {condition}', alpha=0.7)

            plt.title('Value by Category, Group, and Condition')
            plt.ylabel('Value')
            plt.xlabel('Category')
            plt.legend(title='Group - Condition')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        plot_bar_chart(df)

        # Stacked Bar Chart
        def plot_stacked_bar_chart(df):
            pivot_df = df.pivot_table(index=['Group', 'Condition'], columns='Category', values='Value',
                                      aggfunc='sum').fillna(0)
            pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6))

            plt.title('Stacked Bar Chart of Values by Group and Condition')
            plt.ylabel('Value')
            plt.xlabel('Group - Condition')
            plt.xticks(rotation=45)
            plt.legend(title='Category')
            plt.tight_layout()
            plt.show()

        plot_stacked_bar_chart(df)

        def plot_combined_bar_chart_with_counts(df):
            # Combine the counts by summing up 'Most Important' and 'Least Important' for each category
            combined_df = df.groupby(['Group', 'Category'])['Value'].sum().unstack().fillna(0)

            # Create the bar chart
            ax = combined_df.T.plot(kind='bar', figsize=(10, 6))

            # Add the counts on top of each bar
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10),
                            textcoords='offset points')

            # Formatting the chart
            plt.title('Comparison of Categories between Chat and Active Chat')
            plt.ylabel('Total Count')
            plt.xlabel('Category')
            plt.xticks(rotation=45)
            plt.legend(title='Group')
            plt.tight_layout()

            # Show the plot
            plt.show()

        # Call the function to plot the combined bar chart with counts
        plot_combined_bar_chart_with_counts(df)
        print()

    if "plot_question_tornado" in analysis_steps:
        best_ids = data_chat["user_df"][data_chat["user_df"]["study_condition"] == "chat"]["id"].to_list()
        worst_ids = data_chat["user_df"][data_chat["user_df"]["study_condition"] == "active_chat"]["id"].to_list()

        # Replace technical question ids with readable question names
        mappping = {
            "followUp": "Follow Up",
            "whyExplanation": "Why Explanation",
            "counterfactualAnyChange": "Counterfactual",
            "anchor": "Anchors",
            "featureStatistics": "Feature Statistics",
            "top3Features": "Top 3 Features",
            "shapAllFeatures": "Feature Influences",
            "ceterisParibus": "Ceteris Paribus",
            "least3Features": "Least 3 Features",
            "greeting": "Greeting",
            "notXaiMethod": "Not XAI Method",
        }

        data_chat["questions_over_time_df"]["question_id"] = data_chat["questions_over_time_df"]["question_id"].map(
            mappping)

        """best_ids, worst_ids = get_wort_and_best_users(data_chat["user_df"], "final_irt_score_mean", "chat")
        best_ids_active, worst_ids_active = get_wort_and_best_users(data_chat["user_df"], "final_irt_score_mean",
                                                                   "active_chat")"""

        plot_questions_tornado(data_chat["questions_over_time_df"], worst_ids, best_ids, save=False,
                               group1_name="guided-chat", group2_name="chat")

    if "print_correlations" in analysis_steps:
        keep_cols = ["total_learning_time", "exp_instruction_time", "total_exp_time", "failed_checks", "ml_knowledge",
                     "intro_score", "final_avg_confidence", "intro_avg_confidence", "learning_score",
                     "subjective_understanding", "accuracy_over_time", "final_irt_score",
                     "intro_irt_score"]
        for idx, group_name in enumerate(groups_to_compare):
            print(f"Correlation for {group_name}")
            print_correlation_ranking(groups_data[idx], "final_irt_score", keep_cols=keep_cols)

        user_question_freq_chat = make_question_count_df(data_chat["questions_over_time_df"], data_chat["user_df"])
        print_correlation_ranking(user_question_freq_chat, "final_irt_score", "chat")
        print_correlation_ranking(user_question_freq_chat, "final_irt_score", "active_chat")

        # map question ids to str
        data_static_interactive["questions_over_time_df"]["question_id"] = data_static_interactive[
            "questions_over_time_df"]["question_id"].map(id_to_question_mapping)
        user_question_freq_static = make_question_count_df(data_static_interactive["questions_over_time_df"],
                                                           data_static_interactive["user_df"])
        print_correlation_ranking(user_question_freq_static, "final_irt_score", "interactive")

    if "process_mining" in analysis_steps:
        # Process Mining
        best_ids, wors_ids = get_wort_and_best_users(data_chat["user_df"], "chat", "final_irt_score_mean")
        pm = ProcessMining()
        all_ids = best_ids + wors_ids
        pm.create_pm_csv(data_chat["questions_over_time_df"],
                         datapoint_count=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         target_user_ids=all_ids,
                         target_group_name="chat_all")


if __name__ == "__main__":
    groups_to_compare = ["chat", "active_chat", "static", "interactive"]
    main(groups_to_compare)
    """for tuple in [("static", "interactive"),
                  ("static", "chat"),
                  ("static", "active_chat"),
                  ("interactive", "chat"),
                  ("interactive", "active_chat"),
                  ("chat", "active_chat")]:
        main(list(tuple))"""

    """for i in range(10):
        main(["chat", "active_chat"])"""
