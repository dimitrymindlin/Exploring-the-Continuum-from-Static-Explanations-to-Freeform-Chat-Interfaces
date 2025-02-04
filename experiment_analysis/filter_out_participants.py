import json
from datetime import datetime
import pandas as pd
from experiment_analysis.plot_overviews import print_feedback_json
import numpy as np


def filter_by_time(time_df, std_dev_threshold=2):
    """
    Removes outliers based on total_time by excluding participants who are less than x standard deviations from the mean of their study group.
    This version only filters out participants who complete the study too quickly, ignoring those who take longer than average.

    Parameters:
    - time_df: DataFrame with columns 'user_id', 'start_time', 'end_time', 'study_group', and 'total_time'.

    Returns:
    - List of user IDs considered as outliers for being too fast.
    """
    # Calculate mean and standard deviation for total_learning_time by study_group
    mean_time_learning = time_df.groupby('study_group')['total_learning_time'].transform('mean')
    std_time_learning = time_df.groupby('study_group')['total_learning_time'].transform('std')

    # Calculate mean and standard deviation for total_exp_time by study_group
    mean_time_exp = time_df.groupby('study_group')['total_exp_time'].transform('mean')
    std_time_exp = time_df.groupby('study_group')['total_exp_time'].transform('std')

    # Determine if each participant's total_time is less than the lower bound of allowed standard deviation range
    is_too_fast_learning = time_df['total_learning_time'] < (mean_time_learning - std_dev_threshold * std_time_learning)
    is_too_fast_exp = time_df['total_exp_time'] < (mean_time_exp - std_dev_threshold * std_time_exp)

    # Identify too fast participants
    too_fast_participants_learning = time_df[is_too_fast_learning]
    too_fast_participants_exp = time_df[is_too_fast_exp]

    # Get the IDs of users considered too fast
    too_fast_learning_ids = too_fast_participants_learning['user_id'].tolist()
    too_fast_exp_ids = too_fast_participants_exp['user_id'].tolist()

    if len(too_fast_learning_ids) > 0:
        print("Users too fast in learning phase: ", len(too_fast_learning_ids))
    if len(too_fast_exp_ids) > 0:
        print("Users too fast in whole experiment: ", len(too_fast_exp_ids))

    for user_id in too_fast_exp_ids:
        total_time = time_df.loc[time_df["user_id"] == user_id, "total_exp_time"].values[0]
        print(f"       User ID: {user_id}, Experiment Time: {total_time} minutes")

    for user_id in too_fast_learning_ids:
        total_time = time_df.loc[time_df["user_id"] == user_id, "total_learning_time"].values[0]
        print(f"       User ID: {user_id}, Learning Time: {total_time} minutes")

    exclude_user_ids = too_fast_learning_ids + too_fast_exp_ids
    return exclude_user_ids


def remove_outliers_by_attention_check(user_df, user_completed_df):
    # Add new column "failed_checks" to user_df with value 0
    user_df["failed_checks"] = 0

    for user_id in user_completed_df["user_id"]:
        attention_checks = user_completed_df[user_completed_df["user_id"] == user_id]["attention_checks"].values[0]
        for check_id, check_result in attention_checks.items():
            # don't count first check because its comprehension check
            if check_id == "1":
                continue
            if isinstance(check_result['correct'], str):
                if check_result['correct'] != check_result['selected']:
                    user_df.loc[user_df["id"] == user_id, "failed_checks"] += 1
            else:  # is a list
                if check_result['selected'] not in check_result['correct']:
                    user_df.loc[user_df["id"] == user_id, "failed_checks"] += 1

    # Get the IDs of users that failed 2 attention_checks
    failed_attention_check_ids = user_df.loc[user_df["failed_checks"] >= 2, 'id'].tolist()
    print("Users failed attention check: ", len(failed_attention_check_ids))

    # Get the prolific IDs of users that failed 2 attention_checks
    failed_attention_check_prolific_ids = user_df.loc[user_df["failed_checks"] >= 2, 'prolific_id'].tolist()

    # Print the id with their answers to the attention checks
    for user_id in failed_attention_check_ids:
        attention_checks = user_completed_df[user_completed_df["user_id"] == user_id]["attention_checks"].values[0]
        print(f"User ID: {user_id}, Attention Checks: {attention_checks}")

    return failed_attention_check_ids


def filter_by_current_day(user_df):
    # Get today's date
    today = datetime.now().date()

    # Create a copy of user_df
    user_df_copy = user_df.copy()

    # Convert 'created_at' to date only
    user_df_copy.loc[:, 'created_at_day'] = pd.to_datetime(user_df_copy['created_at']).dt.date

    # Filter rows where the date part of 'created_at' is not today
    remove_user_ids = user_df_copy.loc[user_df_copy['created_at_day'] != today, 'id'].tolist()
    return remove_user_ids


def filter_completed_users(user_df):
    incomplete_user_ids = user_df.loc[user_df["completed"] == False, 'id'].tolist()
    print("Users that did not complete the study: ", len(incomplete_user_ids))
    return incomplete_user_ids


def _add_end_time(user_completed_df, prolific_df):
    # Load prolific export and match user_completed with prolific_id to get end_time
    non_prolific_user_ids = []
    non_prolific_prolific_ids = []
    no_end_time = []
    user_completed_df["end_time"] = None
    strange_cases = []
    for index, row in user_completed_df.iterrows():
        prolific_id = row["prolific_id"]
        try:
            # Get end time in the original ISO 8601 format
            end_time_iso = prolific_df[prolific_df["Participant id"] == prolific_id]["Completed at"].values[0]
            start_time_iso = prolific_df[prolific_df["Participant id"] == prolific_id]["Started at"].values[0]
            if end_time_iso is np.nan:
                status = prolific_df[prolific_df["Participant id"] == prolific_id]["Status"].values[0]
                no_end_time.append((prolific_id, status))
                continue

            # Convert to datetime object
            end_time_obj = datetime.strptime(end_time_iso, "%Y-%m-%dT%H:%M:%S.%fZ")
            start_time_obj = datetime.strptime(start_time_iso, "%Y-%m-%dT%H:%M:%S.%fZ")

            if end_time_obj < row["created_at"]:
                prolific_created = start_time_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                prolific_finished = end_time_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                db_created = row["created_at"]
                strange_cases.append((prolific_id, prolific_created, prolific_finished, db_created))
                # Exclude user
                non_prolific_user_ids.append(row["user_id"])
                non_prolific_prolific_ids.append(prolific_id)

            # Format datetime object to the desired string format without 'T'
            end_time_formatted = end_time_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            start_time_formatted = start_time_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        except IndexError:
            non_prolific_user_ids.append(row["user_id"])
            non_prolific_prolific_ids.append(prolific_id)
            continue
        # Save the converted end time back in user_completed_df
        user_completed_df.loc[index, "prolific_end_time"] = end_time_formatted
        user_completed_df.loc[index, "prolific_start_time"] = start_time_formatted

    print("Non prolific users: ", len(non_prolific_user_ids))

    # for missing end time, compute median duration and add to start time
    user_completed_df["prolific_end_time"] = user_completed_df["prolific_end_time"].astype("datetime64")
    user_completed_df["prolific_start_time"] = user_completed_df["prolific_start_time"].astype("datetime64")
    median_duration = user_completed_df["prolific_end_time"] - user_completed_df["prolific_start_time"]
    median_duration = median_duration.median()
    count = 0
    for index, row in user_completed_df.iterrows():
        # Check if completed
        if not row["completed"]:
            continue
        if pd.isnull(row["prolific_end_time"]):
            user_completed_df.loc[index, "prolific_end_time"] = row["prolific_start_time"] + median_duration
            count += 1
    print("Added end time to", count, "users.")
    print("Strange cases:", strange_cases)
    return non_prolific_user_ids


def filter_by_prolific_users(analysis, prolific_file_name):
    # TODO: Include users that are not in prolific export
    prolific_df = pd.read_csv(prolific_file_name)
    include_users_list = ["28389d4b-62ef-411d-a92a-4280cb409cb7",
                          "9c6f2705-0681-4ca8-b56a-d5bc8cb322cb",
                          "45922c7e-d68a-4574-9a53-4794e0139c57",
                          "28389d4b-62ef-411d-a92a-4280cb409cb7",
                          "3715e50e-0a37-4a0e-841c-1b21ba8588f2",
                          "965ce6aa-2ebb-4325-a01a-a06293c9a27a",
                          "1b53a9fe-61f2-4b68-a33e-044f59961928",
                          "e1420868-2337-4935-a730-c77f8275c845"]
    # include_users_list = []
    # Exclude users
    non_prolific_user_ids = _add_end_time(analysis.user_completed_df, prolific_df)
    # Fatal user 8d9f18d9-db90-4ce6-ba44-719e242e51bf
    exclude_users_list = ["a4e2fba8-6e9f-4001-88d2-f109a1f9acc6", "9a84f941-a140-4d47-8766-01dc4e70993e",
                          "8d9f18d9-db90-4ce6-ba44-719e242e51bf",
                          "5fab286a-90b7-4eab-93c0-c033e3667a8b",  # From here, because of knowledge too high
                          "f383429a-10ad-4986-8069-d00ee92865c1",
                          "0d0cac57-1b99-4437-9d52-bcdd4a8d607e",
                          "72f2ade2-58ef-4415-a949-bf4864ac8181",
                          # From here because "Sorry i cannot understand"
                          'ba51a07b-0d9b-4533-a6f6-0c15ebd12191',
                          'e7a5c102-8c1a-4d1e-a9e9-1bd4b0356fdf',
                          'f3ca4b1c-dbcc-455d-a2a5-a3f95dd8d9ae',
                          '7056ef0d-a408-4a28-b539-c08e5823f8ad',
                          'b59e936f-311e-40a1-a793-3fed5cd3e878',
                          'ea960cf7-6cfa-4d43-a48f-d59f8d069b6c',
                          '1b2551b8-d14b-4ea7-9039-c6b2673eadef',
                          '7ba157ae-753b-4169-9d2b-7f05c84dc79e',
                          'c800795b-c050-457d-8af3-29d1e30cf0e0',
                          '24cf145e-c985-4423-becd-bcb98e1373aa',
                          '45535fbf-68a3-4432-bea1-75c78271ecc9',
                          '3c30f681-7215-487c-abfa-61ef56c5b352',
                          '5b0ff77e-3ad2-4aa7-9a77-83ffae20daf3',
                          'f8cb2154-037e-4519-b569-e1648f82a8d1',
                          '61be6d32-5b0c-473c-abcc-bdc768432fd5',
                          'c87eae3b-359c-4723-8c3b-4a7bb334f01c',
                          '91dbbedb-9d53-4a93-8cf2-96f3929efac9',
                          '5cfa78c8-2505-4c3b-b5b1-185b1bf0e0d2']
    non_prolific_user_ids.extend(exclude_users_list)
    non_prolific_user_ids = [user_id for user_id in non_prolific_user_ids if user_id not in include_users_list]
    analysis.update_dfs(non_prolific_user_ids)


def filter_by_work_life_balance(analysis, wlb_users):
    wlb_users = [user for user in wlb_users if user[0] in analysis.user_df["id"].values]
    # Add learning time to wlb_df based on user_id
    wlb_users = [
        (user[0], user[1], analysis.user_df[analysis.user_df["id"] == user[0]]["total_learning_time"].values[0])
        for user in wlb_users]
    # Save wlb users to a csv file
    wlb_df = pd.DataFrame(wlb_users, columns=["user_id", "feedback", "learning_time"])
    wlb_df = wlb_df.sort_values(by="learning_time", ascending=False)
    wlb_df.to_csv("wlb_users.csv", index=False)
    # remove the wlb users from the analysis
    # Print unique remaining wlb users
    print("Unique remaining WLB users: ", wlb_df["user_id"].nunique())
    return wlb_df["user_id"].tolist()


def filter_by_missing_ml_kowledge(analysis):
    # Extract ml_knowledge from profile column
    analysis.user_df["ml_knowledge"] = analysis.user_df["profile"].apply(
        lambda x: int(json.loads(x)["fam_ml_val"]) if json.loads(x)["fam_ml_val"] != '' else None)
    analysis.user_df.drop(columns=["profile"], inplace=True)
    # Filter
    if analysis.user_df["ml_knowledge"].isnull().sum() > 0:
        # print user_ids with missing ml_knowledge values
        users_to_remove = analysis.user_df[analysis.user_df["ml_knowledge"].isnull()]["id"]
        if len(users_to_remove) > 0:
            print("Missing ml_knowledge values: ", analysis.user_df["ml_knowledge"].isnull().sum())
            analysis.update_dfs(users_to_remove)
            print(
                f"Amount of users per study group after removing missing ml_knowledge values: {len(analysis.user_df)}")


def filter_by_negative_times(analysis):
    """
    Checks if there are rows where total_time is negative in the DataFrame.

    :param analysis: The analysis object containing user_df DataFrame.
    """
    remove_user_ids = []
    # Check if there are rows where total_exp_time is negative
    negative_exp_times = analysis.user_df[analysis.user_df["total_exp_time"] < 0][
        ["user_id", "event_start_time", "event_end_time"]]

    if not negative_exp_times.empty:
        remove_user_ids.extend(negative_exp_times["user_id"].tolist())

    # Check if there are rows where exp_instruction_time is negative
    negative_intruct_times = analysis.user_df[analysis.user_df["exp_instruction_time"] < 0][
        ["user_id", "prolific_start_time", "event_start_time"]]
    if not negative_intruct_times.empty:
        remove_user_ids.extend(negative_intruct_times["user_id"].tolist())

    if len(remove_user_ids) > 0:
        print(f"Found {len(remove_user_ids)} users with negative times.")
    return remove_user_ids


def filter_by_broken_variables(analysis):
    """
    Exclude incomplete records where some variables cannot be inferred
    """
    exclude_user_ids = set()

    incomplete_user_ids = filter_completed_users(analysis.user_df)
    exclude_user_ids.update(incomplete_user_ids)

    failed_attention_check_ids = remove_outliers_by_attention_check(analysis.user_df, analysis.user_completed_df)
    exclude_user_ids.update(failed_attention_check_ids)

    too_fast_ids = filter_by_time(analysis.user_df)
    exclude_user_ids.update(too_fast_ids)

    negative_times_ids = filter_by_negative_times(analysis)
    exclude_user_ids.update(negative_times_ids)

    filter_by_missing_ml_kowledge(analysis)

    analysis.update_dfs(exclude_user_ids)


def filter_by_self_defined_attention_check(analysis, wlb_users):
    # Filter by users who used work-life balance as an important variable
    remove_user_ids = filter_by_work_life_balance(analysis, wlb_users)
    analysis.update_dfs(remove_user_ids)
