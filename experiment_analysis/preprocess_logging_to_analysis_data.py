import psycopg2
import pandas as pd
from fuzzywuzzy import fuzz

from experiment_analysis.analysis_data_holder import AnalysisDataHolder
from experiment_analysis.calculations import create_predictions_df
from experiment_analysis.filter_out_participants import filter_by_prolific_users, filter_by_broken_variables
from experiment_analysis.plot_overviews import plot_understanding_over_time
import json

# Load DB configuration from db_config.json
with open('db_config.json', 'r') as config_file:
    config = json.load(config_file)

POSTGRES_USER = config["POSTGRES_USER"]
POSTGRES_PASSWORD = config["POSTGRES_PASSWORD"]
POSTGRES_DB = config["POSTGRES_DB"]
POSTGRES_HOST = config["POSTGRES_HOST"]
prolific_file_name = config["prolific_file_name"]  # Name of prolific export file
result_folder_path = config["result_folder_path"]  # Path to save the results

analysis_steps = ["filter_by_prolific_users",
                  "filter_completed_users",
                  "filter_by_attention_check",
                  "filter_by_time",
                  "remove_users_that_didnt_ask_questions"]


def connect_to_db():
    return psycopg2.connect(
        f"dbname={POSTGRES_DB} user={POSTGRES_USER} host={POSTGRES_HOST} password={POSTGRES_PASSWORD}"
    )


def fetch_data_as_dataframe(query, connection):
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    column_names = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=column_names)


def append_user_data_to_df(df, user_id, study_group, data, column_name="time"):
    if (df['user_id'] == user_id).any():
        # If the user exists, update the existing row
        df.loc[df['user_id'] == user_id, column_name] = data
    else:
        # If the user does not exist, create a new row
        new_row = pd.DataFrame([{"user_id": user_id, "study_group": study_group, column_name: data}])
        df = pd.concat([df, new_row], ignore_index=True)
    return df


def extract_all_feedback(user_df, user_id):
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


def get_study_group_and_events(user_df, event_df, user_id):
    study_group = user_df[user_df["id"] == user_id]["study_group"].values[0]
    user_events = event_df[event_df["user_id"] == user_id]
    return study_group, user_events


def extract_questions(user_events):
    """
    Process the interactive group by extracting user questions over time.
    """
    user_questions_over_time = user_events[
        (user_events["action"] == "question") & (user_events["source"] == "teaching")]

    # Check how many users have not asked any questions (i.e. all cols contain nan)
    if user_questions_over_time.empty:
        return None

    ## UNPACK DETAILS COLUMN
    # Details is a string that contains a dictionary
    user_questions_over_time = user_questions_over_time.copy()
    user_questions_over_time.loc[:, 'details'] = user_questions_over_time['details'].apply(json.loads)
    details_df = user_questions_over_time['details'].apply(pd.Series)
    # replace details column with the new columns
    user_questions_over_time = pd.concat([user_questions_over_time.drop(columns=['details']), details_df], axis=1)
    return user_questions_over_time


def main():
    """merge_prolific_files()
    quit()"""
    conn = connect_to_db()
    analysis = AnalysisDataHolder(user_df=fetch_data_as_dataframe("SELECT * FROM users", conn),
                                  event_df=fetch_data_as_dataframe("SELECT * FROM events", conn),
                                  user_completed_df=fetch_data_as_dataframe("SELECT * FROM user_completed", conn))
    # if "filter_by_prolific_users" in analysis_steps:
    filter_by_prolific_users(analysis, prolific_file_name)
    analysis.create_time_columns()

    print("Found users: ", len(analysis.user_df))
    print("Found events: ", len(analysis.event_df))
    print(analysis.user_df.groupby("study_group").size())

    ### Filtering
    filter_by_broken_variables(analysis)
    print("Amount of users per study group after broken variables filter:")
    print(analysis.user_df.groupby("study_group").size())

    user_questions_over_time_list = []
    initial_test_preds_list = []
    learning_test_preds_list = []
    final_test_preds_list = []
    all_q_list = []
    exclude_user_ids = []
    wlb_users = []
    for user_id in analysis.user_df["id"]:
        study_group, user_events = get_study_group_and_events(analysis.user_df, analysis.event_df, user_id)
        # Create predictions dfs and if there is a user with missing or broken data, exclude them
        # check if user_events is empty
        if user_events.empty:
            exclude_user_ids.append(user_id)
            continue

        # check if source teaching handle next is 10 times
        if len(user_events[user_events["source"] == "teaching"]) < 10:
            exclude_user_ids.append(user_id)
            continue

        if study_group != "static":
            user_questions_over_time_df = extract_questions(user_events)
            if user_questions_over_time_df is not None:
                user_questions_over_time_list.append(user_questions_over_time_df)
            else:
                if "remove_users_that_didnt_ask_questions" in analysis_steps:
                    exclude_user_ids.append(user_id)

        intro_test_preds, preds_learning, final_test_preds, exclude = create_predictions_df(analysis.user_df,
                                                                                            user_events,
                                                                                            exclude_incomplete=True)
        if exclude:
            exclude_user_ids.append(user_id)
            continue
        initial_test_preds_list.append(intro_test_preds)
        learning_test_preds_list.append(preds_learning)
        final_test_preds_list.append(final_test_preds)

        # Check if the user has indicated work-life balance as an important variable for the prediction
        feedback = final_test_preds["feedback"].values
        for f in feedback:
            # Check if the similarity score is above a certain threshold, e.g., 80
            if fuzz.partial_ratio("worklifebalance", f.lower()) > 85 or fuzz.partial_ratio("work life balance",
                                                                                           f.lower()) > 85:
                wlb_users.append((user_id, f, study_group))

        all_questionnaires_df = extract_all_feedback(analysis.user_df, user_id)
        all_q_list.append(all_questionnaires_df)

    # Update analysis dfs and exclude users
    print(f"Users excluded: {len(exclude_user_ids)}")
    analysis.update_dfs(exclude_user_ids)
    analysis.add_self_assessment_value_column()

    print("Amount of users per study group after broken variables filter:")
    print(analysis.user_df.groupby("study_group").size())

    # Save wlb_users to a csv file for manual analysis
    wlb_users_df = pd.DataFrame(wlb_users, columns=["user_id", "feedback", "study_group"])
    wlb_users_df.to_csv(f"wlb_users_{prolific_file_name}.csv", index=False)
    # get wlb user ids and exclude them
    wlb_user_ids = list(set(wlb_users_df["user_id"].values))
    analysis.update_dfs(wlb_user_ids)

    print("Amount of users per study group after first filters:")
    print(analysis.user_df.groupby("study_group").size())

    ### Add Additional Columns to analysis.user_df
    # Merge final_q_feedback_list to analysis.user_df on user_id
    analysis.user_df = analysis.user_df.merge(pd.concat(all_q_list), left_on="id", right_on="user_id", how="left")
    initial_test_preds_df = pd.concat(initial_test_preds_list)
    learning_test_preds_df = pd.concat(learning_test_preds_list)
    analysis.add_initial_test_preds_df(initial_test_preds_df)
    analysis.add_learning_test_preds_df(learning_test_preds_df)
    analysis.add_final_test_preds_df(pd.concat(final_test_preds_list))
    analysis.add_questions_over_time_df(pd.concat(user_questions_over_time_list))

    # Update all dfs
    analysis.update_dfs()

    print(analysis.user_df.loc[analysis.user_df['id'] == "5470e036-7300-4de0-bd37-088a0a7816e5"])

    assert len(analysis.user_df) == len(analysis.initial_test_preds_df['user_id'].unique())
    assert len(analysis.user_df) == len(analysis.learning_test_preds_df['user_id'].unique())

    user_accuracy_over_time_df = plot_understanding_over_time(analysis.learning_test_preds_df, analysis)
    assert len(analysis.user_df) == len(user_accuracy_over_time_df['user_id'].unique())

    # Extract questionnaires into columns
    # analysis.user_df = extract_questionnaires(analysis.user_df)

    # Turn mL knowledge to int
    analysis.user_df["ml_knowledge"] = analysis.user_df["ml_knowledge"].astype(int)
    print(analysis.user_df["ml_knowledge"].value_counts())
    # Print how many > 3 per study group
    print(analysis.user_df[analysis.user_df["ml_knowledge"] > 4].groupby("study_group").size())

    print("Amount of users per study group after All filters:")
    print(analysis.user_df.groupby("study_group").size())

    # Delete sentivive information from user_completed_df
    ## Delete prolific_id
    analysis.user_completed_df.drop(columns=["prolific_id"], inplace=True)

    # Save the dfs to csv
    analysis.user_df.to_csv(result_folder_path + "user_df.csv", index=False)
    analysis.event_df.to_csv(result_folder_path + "event_df.csv", index=False)
    analysis.user_completed_df.to_csv(result_folder_path + "user_completed_df.csv", index=False)
    analysis.initial_test_preds_df.to_csv(result_folder_path + "initial_test_preds_df.csv", index=False)
    analysis.learning_test_preds_df.to_csv(result_folder_path + "learning_test_preds_df.csv", index=False)
    analysis.final_test_preds_df.to_csv(result_folder_path + "final_test_preds_df.csv", index=False)
    analysis.questions_over_time_df.to_csv(result_folder_path + "questions_over_time_df.csv", index=False)
    user_accuracy_over_time_df.to_csv(result_folder_path + "user_accuracy_over_time_df.csv", index=False)

    # Load final_chat_user_ids.csv
    final_chat_user_ids = pd.read_csv("final_chat_user_ids.csv")
    # Filter analysis.user_df by final_chat_user_ids
    analysis.user_df = analysis.user_df[analysis.user_df["id"].isin(final_chat_user_ids["id"])]

    # Get demographics of the users
    user_prolific_ids = analysis.user_df["prolific_id"].values
    prolific_df = pd.read_csv(prolific_file_name)
    prolific_df = prolific_df[prolific_df["Participant id"].isin(user_prolific_ids)]
    # Get Age statistics
    # ignore revoked from analysis for now
    prolific_df = prolific_df[prolific_df["Age"] != "CONSENT_REVOKED"]
    # turn age to int
    prolific_df["Age"] = prolific_df["Age"].astype(int)
    print(prolific_df["Age"].describe())
    print()
    # Get Sex statistics
    print(prolific_df["Sex"].value_counts())
    print(prolific_df["Nationality"].value_counts())
    # Turn mL knowledge to int
    analysis.user_df["ml_knowledge"] = analysis.user_df["ml_knowledge"].astype(int)
    print(analysis.user_df["ml_knowledge"].value_counts())
    # Print ml knowledge categories per study group
    print(analysis.user_df.groupby(["study_group", "ml_knowledge"]).size())

    print()

    # Get top 10 people based on final score with their prolific id
    # print(analysis.user_df[["prolific_id", "final_score"]].sort_values("final_score", ascending=False).head(10))


if __name__ == "__main__":
    main()
