import json

import pandas as pd


def calculate_user_score_from_preds(user_predictions_df):
    user_score = 0
    for _, row in user_predictions_df.iterrows():
        try:
            if row['prediction'].lower() == row['true_label'].lower():
                user_score += 1
        except AttributeError:
            # This is hacky. For some reason for one participant true label is not logged... Get it like this.
            correct_prediction_array = pd.read_csv("correct_predictions_final_test.csv").to_numpy()
            prediction_id = row["datapoint_count"]
            if row["prediction"].lower() == correct_prediction_array[prediction_id][0]:
                user_score += 1

    return int(user_score)


def calculate_avg_user_confidence_from_preds(user_predictions_df):
    """
    Calculate the average confidence of a user based on their predictions.
    """
    user_confidence = 0
    for _, row in user_predictions_df.iterrows():
        user_confidence += int(row['confidence_level'])

    user_confidence /= len(user_predictions_df)
    return user_confidence


def load_json_details(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}  # Return empty dict if JSON is malformed


def normalize_details(df):
    details = df['details'].apply(load_json_details)
    df = pd.concat([df.drop(['details'], axis=1), details.apply(pd.Series)], axis=1)
    df['accuracy'] = df.apply(lambda row: 'Correct' if row['prediction'] == row['true_label'] else 'Wrong', axis=1)
    return df


def process_and_remove_duplicates(df, key_columns):
    if 'feedback' in df.columns:
        key_columns.append('feedback')
    unique_indices = df.drop_duplicates(subset=key_columns).index
    return df.loc[unique_indices]


def calculate_score_and_confidence(predictions_df, phase="final"):
    if predictions_df.empty:
        return 0, 0
    score = calculate_user_score_from_preds(predictions_df)
    if not phase == "learning":
        confidence = calculate_avg_user_confidence_from_preds(predictions_df)
    else:
        confidence = 0
    return score, confidence


def update_user_df(user_df, user_id, score_intro, score_final, confidence_intro, confidence_final, score_learning):
    conditions = user_df["id"] == user_id
    user_df.loc[conditions, "final_score"] = score_final
    user_df.loc[conditions, "intro_score"] = score_intro
    user_df.loc[conditions, "final_avg_confidence"] = confidence_intro
    user_df.loc[conditions, "intro_avg_confidence"] = confidence_final
    user_df.loc[conditions, "learning_score"] = score_learning


def create_predictions_df(user_df, user_events, exclude_incomplete=False):
    exclude = False
    # Filter predictions by source and action
    predictions = (user_events["action"] == "user_prediction")
    predictions_final_test = user_events[predictions & (user_events["source"] == "final-test")].copy()
    predictions_intro_test = user_events[predictions & (user_events["source"] == "intro-test")].copy()
    predictions_learning_test = user_events[predictions & (user_events["source"] == "test")].copy()

    # Normalize 'details' dictionary
    predictions_final_test = normalize_details(predictions_final_test)
    predictions_intro_test = normalize_details(predictions_intro_test)
    predictions_learning_test = normalize_details(predictions_learning_test)

    # Remove duplicates based on confidence level and accuracy
    key_columns = ['datapoint_count', 'confidence_level', 'accuracy']
    predictions_final_test = process_and_remove_duplicates(predictions_final_test, key_columns.copy())
    predictions_intro_test = process_and_remove_duplicates(predictions_intro_test, key_columns.copy())
    predictions_learning_test = process_and_remove_duplicates(predictions_learning_test,
                                                              ["datapoint_count", "accuracy"])

    if exclude_incomplete:
        if len(predictions_final_test) != 10 or len(predictions_intro_test) != 10 or len(predictions_learning_test) < 10:
            return None, None, None, True

    # Calculate scores and confidence
    score_intro, confidence_intro = calculate_score_and_confidence(predictions_intro_test)
    score_final, confidence_final = calculate_score_and_confidence(predictions_final_test)
    score_learning, _ = calculate_score_and_confidence(predictions_learning_test, phase="learning")

    # Update user_df with scores and confidences
    user_id = user_events['user_id'].unique()[0]
    update_user_df(user_df, user_id, score_intro, score_final, confidence_intro, confidence_final, score_learning)

    return predictions_intro_test, predictions_learning_test, predictions_final_test, exclude
