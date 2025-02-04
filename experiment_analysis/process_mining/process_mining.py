import math

import numpy as np
import pandas as pd
from datetime import datetime

question_text = {23: "Most Important Features",
                 27: "Least Important Features",
                 24: "Feature Attributions",
                 7: "Counterfactuals",
                 11: "Anchors",
                 25: "Ceteris Paribus",
                 13: "Feature Ranges"}


def get_timestamp(row):
    """
    Convert the 'created_at' field of the given row to a formatted timestamp without microseconds.

    Parameters:
    row (dict): A dictionary containing the 'created_at' field as a string in the format 'YYYY-MM-DD HH:MM:SS.microseconds'.

    Returns:
    str: The formatted timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    created_at_str = row["created_at"]
    created_at_dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S.%f")
    timestamp = created_at_dt.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp

class ProcessMining:
    def __init__(self):
        self.pm_event_df = None

    def _filter_by_score(self, user_scores_df, filter_by_score):
        user_scores_df = user_scores_df.reset_index()
        user_scores_df.rename(columns={"index": "user_id"}, inplace=True)
        final_scores_dict = user_scores_df.set_index("user_id")["Aggr. Final Score"].to_dict()

        if filter_by_score == "best":
            # Only keep users with score > 3-5, directly assign the filtered DataFrame
            user_scores_df = user_scores_df[user_scores_df["Aggr. Final Score"] >= 5]
        elif filter_by_score == "worst":
            # Only keep users with score > 0-2, directly assign the filtered DataFrame
            user_scores_df = user_scores_df[user_scores_df["Aggr. Final Score"] <= 5]
        else:
            raise ValueError("filter_by_score must be 'best' or 'worst'")

        return user_scores_df, final_scores_dict

    def create_pm_csv(self,
                      questions_over_time_df,
                      datapoint_count=[5],
                      target_user_ids=None,
                      target_group_name="none"):
        """
        Preprocesses experiment event df into a format suitable for process mining.
        Format should be table of [CaseID, Activity, Timestamp]
        :param event_df: DataFrame with event data
        :param user_scores_df: DataFrame with user scores, used to filter users by score
        :param datapoint_count: Datapoint count to filter by datapoint (conversation turn)
        :param target_user_ids: List of user ids to filter by
        """
        if target_user_ids is not None:
            questions_df = questions_over_time_df[
                questions_over_time_df["user_id"].isin(target_user_ids)]
        else:
            raise ValueError("Best or worst user ids must be provided.")

        pm_event_df = pd.DataFrame(columns=["Case id", "Activity", "Timestamp"])

        process_mining_rows = []  # List to collect new rows for process mining

        for user_id in questions_df["user_id"].unique():
            user_questions = questions_df[questions_df["user_id"] == user_id]
            user_questions = user_questions.sort_values("created_at")

            # Collect rows for process mining
            for index, row in user_questions.iterrows():
                # Filter by datapoint count
                if int(row["datapoint_count"]) not in datapoint_count:
                    continue
                question_id = row["question_id"]
                if "chat" not in target_group_name:
                    question = question_text[question_id]
                else:
                    question = question_id
                case_id = row["user_id"] + "_" + str(row["datapoint_count"])
                timestamp = get_timestamp(row)
                process_mining_rows.append({"Case id": case_id, "Activity": question, "Timestamp": timestamp})
        # Append new rows to pm_event_df in one operation
        if process_mining_rows:
            pm_event_df = pd.DataFrame(process_mining_rows) if self.pm_event_df is None else pd.concat(
                [self.pm_event_df, pd.DataFrame(process_mining_rows)], ignore_index=True)

        self.pm_event_df = pm_event_df
        # Print amount of nan values
        print(self.pm_event_df.isna().sum())
        # Fill nan with "Unknown"
        self.pm_event_df.fillna("Unknown", inplace=True)
        # Save as csv file with header
        self.pm_event_df.to_csv(f"process_mining/pm_{datapoint_count}_{target_group_name}.csv", index=False, header=True)

