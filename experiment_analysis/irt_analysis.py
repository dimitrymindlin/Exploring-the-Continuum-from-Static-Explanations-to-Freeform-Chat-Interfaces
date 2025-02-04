import numpy as np
import pandas as pd
from py_irt.models import OneParamLog
from py_irt.dataset import Dataset
from scipy.optimize import minimize


def logistic_function(theta, b):
    """Calculate the probability of a correct response."""
    return 1 / (1 + np.exp(-(theta - b)))


def negative_log_likelihood(theta, responses, difficulties):
    """Calculate the negative log likelihood for the responses given an ability."""
    probabilities = logistic_function(theta, difficulties)
    log_likelihood = np.sum(
        responses * np.log(probabilities + 1e-9) + (1 - responses) * np.log(1 - probabilities + 1e-9))
    return -log_likelihood


def estimate_ability_for_user(responses, difficulties):
    """Estimate the ability of a single user."""
    initial_guess = 0
    result = minimize(negative_log_likelihood, initial_guess, args=(responses, difficulties), method='BFGS')
    return result.x[0]


def estimate_abilities(data, difficulties):
    """Estimate abilities for all users in the dataset."""
    abilities = []
    for index, row in data.iterrows():
        estimated_ability = estimate_ability_for_user(row[1:], difficulties)  # Skip user_id column
        abilities.append({'id': row['user_id'], 'estimated_ability': estimated_ability})
    return pd.DataFrame(abilities)


def prepare_data(data):
    """Prepare data by pivoting and encoding correct responses."""
    data = data.groupby(['user_id', 'datapoint_count']).agg({'accuracy': 'max'}).reset_index()
    wide_df = data.pivot(index='user_id', columns='datapoint_count', values='accuracy')
    wide_df = wide_df.applymap(lambda x: 1 if x == 'Correct' else 0)
    wide_df.columns = [f'item_{i}' for i in range(1, len(wide_df.columns) + 1)]
    wide_df.reset_index(inplace=True)
    return wide_df


def train_irt_model(df):
    """Train the IRT model on training data."""
    dataset = Dataset.from_pandas(df, subject_column="user_id", item_columns=df.columns[1:].tolist())
    trainer = OneParamLog.train(dataset)
    return trainer, trainer.irt_model.export()['ability']


def apply_irt_model(user_data, difficulties):
    """Apply trained IRT model to estimate user abilities."""
    prepared_data = prepare_data(user_data)
    estimated_abilities = estimate_abilities(prepared_data, difficulties)
    return estimated_abilities


def get_irt_score_for_users(user_predictions_df, score_name, trainer=None):
    # Prepare the data
    prepared_data = prepare_data(user_predictions_df)
    real_estimates = None
    if trainer is None:
        # Train the model
        trainer = train_irt_model(prepared_data)
        real_estimates = trainer.irt_model.export()['diff']

    # Extract item difficulties from the trained model
    item_difficulties = trainer.irt_model.export()['diff']

    # Estimate abilities using the item difficulties
    estimated_abilities_df = estimate_abilities(prepared_data, item_difficulties)
    """if real_estimates is not None:
        # Check if real estimates are close to the estimated ones
        assert np.allclose(real_estimates, item_difficulties, atol=1e-2)"""

    # Rename the columns
    estimated_abilities_df.columns = ['id', score_name]

    return estimated_abilities_df, trainer


def get_irt_trainer_for_users_from_multiple_dfs(user_predictions_df_list, col_name):
    user_predictions_list = []
    user_indices = []
    for user_predictions_df in user_predictions_df_list:
        wide_df = prepare_data(user_predictions_df)
        user_predictions_list.append(wide_df)
        user_indices.append(wide_df["user_id"])
    # Prepare the data
    prepared_data = pd.concat(user_predictions_list)
    trainer, scores = train_irt_model(prepared_data)

    # Make mapping from list of scores to list of user_ids
    user_ids_list = []
    for user_index in user_indices:
        user_ids_list.extend(user_index)
    user_scores_mapping = pd.DataFrame({'id': user_ids_list, col_name: scores})

    return trainer, user_scores_mapping
