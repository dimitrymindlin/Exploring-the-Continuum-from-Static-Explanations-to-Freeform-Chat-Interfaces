import ast

import pandas as pd


def __get_profile_dict(user_df, user_id):
    """Get user profile dictionary from user_df"""
    profile_string = user_df[user_df["id"] == user_id]['profile'].values[0]
    # Convert string to dictionary
    profile_dict = ast.literal_eval(profile_string)
    return profile_dict


def prepare_correlation_df(user_df, event_df, understanding_df):
    """Prepare a dataframe for correlation analysis between user understanding and other variables"""
    correlation_dict = {}

    for user_id in user_df["id"].unique():
        # 1. Get user information from user_df json
        profile_dict = __get_profile_dict(user_df, user_id)
        ml_experience = profile_dict["fam_ml_val"]
        domain_experience = profile_dict["fam_domain_val"]
        english_proficiency = profile_dict["english_speaking_level"]
        age = profile_dict["age"]
        gender = profile_dict['gender']

        # 2. Get qeuestions behavior

        # 3. Get time spent on experiment

        # Get amount of questions asked
        user_events = event_df[event_df["user_id"] == user_id]
        user_understanding = understanding_df[understanding_df["user_id"] == user_id]
        user_understanding = user_understanding["understanding"].values[0]
        user_events = user_events.sort_values("created_at")
        user_events = user_events.reset_index(drop=True)
        user_events["user_understanding"] = user_understanding
        user_events["user_id"] = user_id
        try:
            correlation_df = pd.concat([correlation_df, user_events])
        except NameError:
            correlation_df = user_events
