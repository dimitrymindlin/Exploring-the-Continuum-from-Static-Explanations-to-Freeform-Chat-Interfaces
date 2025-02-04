import json

import pandas as pd

from experiment_analysis.plot_overviews import plot_time_boxplots


class AnalysisDataHolder:
    def __init__(self, user_df, event_df, user_completed_df):
        self.user_df = user_df
        self.event_df = event_df
        self.user_completed_df = user_completed_df
        self.initial_test_preds_df = None
        self.learning_test_preds_df = None
        self.final_test_preds_df = None
        self.questions_over_time_df = None

    def update_dfs(self, user_ids_to_remove=None):
        if user_ids_to_remove is None:
            # Update based on ids in self.user_df
            user_ids_to_keep = self.user_df["id"]
            self.event_df = self.event_df[self.event_df["user_id"].isin(user_ids_to_keep)]
            self.user_completed_df = self.user_completed_df[self.user_completed_df["user_id"].isin(user_ids_to_keep)]
            if self.initial_test_preds_df is not None:
                self.initial_test_preds_df = self.initial_test_preds_df[
                    self.initial_test_preds_df["user_id"].isin(user_ids_to_keep)]
            if self.learning_test_preds_df is not None:
                self.learning_test_preds_df = self.learning_test_preds_df[
                    self.learning_test_preds_df["user_id"].isin(user_ids_to_keep)]
            if self.final_test_preds_df is not None:
                self.final_test_preds_df = self.final_test_preds_df[
                    self.final_test_preds_df["user_id"].isin(user_ids_to_keep)]
            if self.questions_over_time_df is not None:
                self.questions_over_time_df = self.questions_over_time_df[
                    self.questions_over_time_df["user_id"].isin(user_ids_to_keep)]
            return
        self.user_df = self.user_df[~self.user_df["id"].isin(user_ids_to_remove)]
        self.event_df = self.event_df[~self.event_df["user_id"].isin(user_ids_to_remove)]
        self.user_completed_df = self.user_completed_df[~self.user_completed_df["user_id"].isin(user_ids_to_remove)]
        if self.initial_test_preds_df is not None:
            self.initial_test_preds_df = self.initial_test_preds_df[
                ~self.initial_test_preds_df["user_id"].isin(user_ids_to_remove)]
        if self.learning_test_preds_df is not None:
            self.learning_test_preds_df = self.learning_test_preds_df[
                ~self.learning_test_preds_df["user_id"].isin(user_ids_to_remove)]
        if self.final_test_preds_df is not None:
            self.final_test_preds_df = self.final_test_preds_df[
                ~self.final_test_preds_df["user_id"].isin(user_ids_to_remove)]
        if self.questions_over_time_df is not None:
            self.questions_over_time_df = self.questions_over_time_df[
                ~self.questions_over_time_df["user_id"].isin(user_ids_to_remove)]

    def add_initial_test_preds_df(self, initial_test_preds_df):
        self.initial_test_preds_df = initial_test_preds_df

    def add_learning_test_preds_df(self, learning_test_preds_df):
        self.learning_test_preds_df = learning_test_preds_df

    def add_final_test_preds_df(self, final_test_preds_df):
        self.final_test_preds_df = final_test_preds_df

    def add_questions_over_time_df(self, questions_over_time_df):
        """
        Adds the questions_over_time_df to the AnalysisDataHolder.
        Extracts the 'datapoint_count', 'question', and 'question_id' columns from the 'details' column.
        """
        """questions_over_time_df['details'] = questions_over_time_df['details'].apply(json.loads)
        details_df = questions_over_time_df['details'].apply(pd.Series)
        # Concatenate the original DataFrame with the new columns
        questions_over_time_df = pd.concat([questions_over_time_df.drop(columns=['details']), details_df], axis=1)"""
        self.questions_over_time_df = questions_over_time_df

    def create_time_columns(self):
        """
        Prepares a DataFrame with start and end times, and total time spent, for each user.

        :param user_df: DataFrame containing user data, including 'id' and 'created_at' columns.
        :param event_df: DataFrame containing event data, with 'user_id' and 'created_at' columns.
        :return: DataFrame with user IDs, start and end times, study group, and total time spent.
        """

        # Prepare initial time_df
        time_df = pd.DataFrame({
            "user_id": self.user_df["id"],
        })

        # Get start event for each user from event_df
        start_time_df = self.event_df.groupby("user_id")["created_at"].min()
        time_df["event_start_time"] = time_df["user_id"].map(start_time_df)

        # Get Experiment end time and start time
        experiemnt_end_time_df = self.user_completed_df.groupby("user_id")["prolific_end_time"].max()
        time_df["prolific_end_time"] = time_df["user_id"].map(experiemnt_end_time_df)
        experiment_start_time_df = self.user_completed_df.groupby("user_id")["prolific_start_time"].min()
        time_df["prolific_start_time"] = time_df["user_id"].map(experiment_start_time_df)

        # Get end time for each user from event_df
        end_time_df = self.event_df.groupby("user_id")["created_at"].max()
        time_df["event_end_time"] = time_df["user_id"].map(end_time_df)

        # Ensure datetime format
        time_df["prolific_start_time"] = pd.to_datetime(time_df["prolific_start_time"])
        time_df["event_start_time"] = pd.to_datetime(time_df["event_start_time"])
        time_df["event_end_time"] = pd.to_datetime(time_df["event_end_time"])
        time_df["prolific_end_time"] = pd.to_datetime(time_df["prolific_end_time"])

        # Calculate total_time in minutes
        time_df["total_learning_time"] = (time_df["event_end_time"] - time_df[
            "event_start_time"]).dt.total_seconds() / 60

        # Total time spent in the experiment instruction (prolific start - first event)
        time_df["exp_instruction_time"] = (time_df["event_start_time"] - time_df[
            "prolific_start_time"]).dt.total_seconds() / 60

        time_df["total_exp_time"] = (time_df["event_end_time"] - time_df[
            "event_start_time"]).dt.total_seconds() / 60

        self.user_df = self.user_df.merge(time_df, left_on="id", right_on="user_id", how="left")

    def add_self_assessment_value_column(self):
        # The self_assessment is in "questionnaires" column in user_df
        remove_users = []
        try:
            self.user_df["self_assessment_understanding"] = self.user_df["questionnaires"].apply(
                lambda x: json.loads(x[2]) if isinstance(x, list) and len(x) > 2 else None)
        except KeyError:
            self_assessment_df = self.user_df[["id", "self_assessment_q"]]
            self.user_df["self_assessment_understanding"] = list(self_assessment_df["self_assessment_q"].apply(lambda x: json.loads(x) if x is not None else None))

        # Calculate the self-assessment value by aggregating the "answers" list
        """
        'I understand the important attributes for a decision.',
        'I cannot distinguish between the possible prediction.',
        'I understand how the Machine Learning model works.',
        'Select "Strongly Disagree" for this selection.',
        'I did not understand how the Machine Learning model makes decisions."""

        for user_id, row in self.user_df.iterrows():
            if row["self_assessment_understanding"] is not None:
                try:
                    answers = self.user_df.loc[user_id, "self_assessment_understanding"]['self_assessment']['answers']
                except KeyError:
                    # How come some users don't have self-assessment answers?
                    remove_users.append(row["id"])
                # TODO: How to handle people who didn't pass attention check here?
                understanding_value = answers[0] + -1 * answers[1] + answers[2] + -1 * answers[4]  # Range: -8 to 8
                self.user_df.loc[user_id, "subjective_understanding"] = understanding_value
        if len(remove_users) > 0:
            print(f"Removing users since they don't have self-assessment: {len(remove_users)}")
            self.update_dfs(remove_users)
