from statistics import median

import pandas as pd
import json

from collections import defaultdict

interactive_question_ids = [23, 27, 24, 7, 11, 25, 13]
static_explanation_ids = [23, 27, 24, 7, 11]


def ranking_method(rankings, method='borda'):
    """
    Calculate rankings based on the specified method ('borda' or 'median'), and return ranking scores.

    Args:
        rankings (list of lists of tuples): Each inner list is a ranking for a participant.
            Each tuple contains (question_id, rank).
        method (str): The ranking method to use ('borda' or 'median').

    Returns:
        list: Sorted list of tuples, each containing the question_id and its score/rank.
    """

    if method == 'borda':
        score_dict = defaultdict(int)
        # Find the highest rank that will be used to calculate points
        max_rank = max(rank for ranking in rankings for _, rank in ranking if rank != "")

        # Calculate Borda count for each ranking list
        for ranking in rankings:
            for question_id, rank in ranking:
                if rank != "":
                    # The points are now max_rank + 1 - rank because lower rank means higher importance
                    score_dict[question_id] += (max_rank + 1 - rank)

        # Sort the question IDs based on their Borda score in descending order
        overall_ranking = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)

    elif method == 'median':
        # Initialize a dictionary to collect all ranks for each question
        rank_dict = defaultdict(list)

        # Aggregate ranks for each question
        for ranking in rankings:
            for question_id, rank in ranking:
                if rank != "":
                    rank_dict[question_id].append(int(rank))

        # Calculate median rank for each question
        median_ranks = {question_id: median(ranks) for question_id, ranks in rank_dict.items()}

        # Sort the question IDs based on their median rank in ascending order (lower rank is better)
        overall_ranking = sorted(median_ranks.items(), key=lambda item: item[1])

    else:
        raise ValueError("Unsupported ranking method. Choose 'borda' or 'median'.")

    # Return the question_ids along with their scores/ranks sorted by the selected method
    return overall_ranking


def get_interactive_ranking(user_df,
                            filter_ids=None):
    # Get questionnaires
    questionnaires = user_df[['id', 'ranking_q']]
    # Normalize the questionnaires column
    questionnaires = pd.concat([questionnaires.drop(['ranking_q'], axis=1),
                                questionnaires['ranking_q'].apply(pd.Series)], axis=1)

    if filter_ids is not None:
        questionnaires = questionnaires[questionnaires['id'].isin(filter_ids)]

    questionnaires.columns = ['id', 'questionnaires']
    # Get the interactive ranking (col 1)
    ranking_list = list(questionnaires["questionnaires"])
    all_rankings = []
    for ranking_string in ranking_list:
        ranking_dict = json.loads(ranking_string)
        ranking = ranking_dict['question_ranking']["answers"]
        # Zip the question IDs with the ranking
        ranking = list(zip(interactive_question_ids, ranking))
        all_rankings.append(ranking)
    overall_ranking = ranking_method(all_rankings, method='median')
    print(overall_ranking)
    # Replace the question IDs with the corresponding question text
    question_text = {23: "Most Important Features",
                     27: "Least Important Features",
                     24: "Feature Attributions",
                     7: "Counterfactuals",
                     11: "Anchors",
                     25: "Ceteris Paribus",
                     13: "Feature Ranges"}
    overall_ranking = [question_text[question_id] for question_id, _ in overall_ranking]
    print(overall_ranking)
    return overall_ranking


def get_static_ranking(user_df):
    # Get questionnaires
    questionnaires = user_df[['id', 'questionnaires']]
    # Normalize the questionnaires column
    questionnaires = pd.concat([questionnaires.drop(['questionnaires'], axis=1),
                                questionnaires['questionnaires'].apply(pd.Series)], axis=1)
    # Get the interactive ranking (col 1)
    ranking_list = list(questionnaires[1].apply(lambda x: json.loads(x)))
    all_rankings = []
    for ranking_dict in ranking_list:
        ranking = ranking_dict['question_ranking']["answers"]
        # Zip the question IDs with the ranking
        ranking = list(zip(static_explanation_ids, ranking))
        all_rankings.append(ranking)
    overall_ranking = ranking_method(all_rankings, method='median')
    print(overall_ranking)
    # Replace the question IDs with the corresponding question text
    question_text = {23: "Feature Attributions", 27: "Counterfactuals", 24: "Anchors", 7: "Ceteris Paribus",
                     11: "Feature Ranges"}
    overall_ranking = [(question_text[question_id]) for question_id in overall_ranking]
    print(overall_ranking)
    return overall_ranking


def extract_questionnaires(user_df):
    # Get questionnaires
    questionnaires = user_df[['id', 'questionnaires']]
    # Normalize the questionnaires column
    questionnaires = pd.concat([questionnaires.drop(['questionnaires'], axis=1),
                                questionnaires['questionnaires'].apply(pd.Series)], axis=1)
    # drop empty columns
    questionnaires.drop([0], axis=1, inplace=True)
    # drop 4th column
    questionnaires.drop([3], axis=1, inplace=True)
    # Change column names
    questionnaires.columns = ['id', 'ranking_q', 'self_assessment_q', 'exit_q']
    # Apply to user_df and drop id column
    user_df = user_df.merge(questionnaires, on='id', how='left')
    return user_df
