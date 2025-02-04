import ast
import json

import pandas as pd


def extract_button_feedback(user_df, user_id):
    try:
        questionnaires = user_df[user_df["id"] == user_id]["questionnaires"].values[0]
        feedback_dict = {"user_id": user_id}
        for questionnaire in questionnaires:
            if isinstance(questionnaire, dict):
                continue
            questionnaire = json.loads(questionnaire)
            for key, value in questionnaire.items():
                feedback_dict[f"{key}"] = value['questions']
                feedback_dict[f"{key}_answers"] = value['answers']
    except (KeyError, IndexError):
        print(f"{user_id} did not complete the questionnaires.")

    # Create a DataFrame from the feedback dictionary
    feedback_df = pd.DataFrame([feedback_dict])
    return feedback_df


def get_button_feedback(event_df, filter_button_type=None):
    """
    Filter the events for feedback button clicks and collect feedback per question and user
    :param event_df: DataFrame with events
    :param filter_button_type: (str) Filter for a button type (e.g. "dislike" or "like")
    """

    question_feedback_dict = {}
    user_feedback_dict = {}

    # Filter for feedback button clicks
    feedback_button_click_rows = event_df[event_df["action"] == "feedback"].copy()
    feedback_button_click_rows['details'] = feedback_button_click_rows['details'].apply(json.loads)
    details_expanded = feedback_button_click_rows['details'].apply(pd.Series)
    feedback_button_click_rows = pd.concat([feedback_button_click_rows.drop(columns=['details']), details_expanded],
                                           axis=1)

    if filter_button_type is not None:
        feedback_button_click_rows = feedback_button_click_rows[
            feedback_button_click_rows["buttonType"] == filter_button_type]

    # Collect feedback per question and user
    for index, row in feedback_button_click_rows.iterrows():
        try:
            question = row["question_id"]
        except KeyError:  # Datapoint like forgot to log question_id...
            continue

        feedback = row["user_comment"]
        user_id = row["user_id"]

        # Add feedback to question_feedback_dict
        if question not in question_feedback_dict:
            question_feedback_dict[question] = []
        question_feedback_dict[question].append(feedback)

        # Add feedback to user_feedback_dict
        if user_id not in user_feedback_dict:
            user_feedback_dict[user_id] = []
        user_feedback_dict[user_id].append(feedback)

    # Print unique user_id s
    print("Unique user_ids: ", len(feedback_button_click_rows["user_id"].unique()))

    return question_feedback_dict, user_feedback_dict
