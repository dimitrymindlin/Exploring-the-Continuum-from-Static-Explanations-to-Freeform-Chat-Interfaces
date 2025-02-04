"""Executes actions for parsed canonical utterances.

This file implements routines to take actions in the conversation, returning
outputs to the user. Actions in the conversation are called `operations` and
are things like running an explanation or performing filtering.
"""
import numpy as np
from flask import Flask

from explain.actions.explanation import explain_cfe, explain_local_feature_importances, explain_cfe_by_given_features, \
    explain_anchor_changeable_attributes_without_effect, explain_feature_statistic, explain_feature_importances_as_plot, \
    explain_ceteris_paribus
from explain.actions.filter import filter_operation
from explain.conversation import Conversation
from explain.actions.get_action_functions import get_all_action_functions_map
from explain.actions.prediction_likelihood import predict_likelihood

app = Flask(__name__)


def run_action(conversation: Conversation,
               parse_tree,
               parsed_string: str,
               actions=get_all_action_functions_map(),
               build_temp_dataset: bool = True) -> str:
    """Runs the action and updates the conversation object

    Arguments:
        build_temp_dataset: Whether to use the temporary dataset stored in the conversation
                            or to rebuild the temporary dataset from scratch.
        actions: The set of avaliable actions
        parsed_string: The grammatical text
        conversation: The conversation object, see `conversation.py`
        parse_tree: The parse tree of the canonical utterance. Note, currently, this is not used,
                    and we compute the actions from the parsed text.
    """
    if parse_tree:
        pretty_parse_tree = parse_tree.pretty()
        app.logger.info(f'Parse tree {pretty_parse_tree}')

    return_statement = ''

    # Will rebuilt the temporary dataset if requested (i.e, for filtering from scratch)
    if build_temp_dataset:
        conversation.build_temp_dataset()

    parsed_text = parsed_string.split(' ')
    is_or = False

    for i, p_text in enumerate(parsed_text):
        if parsed_text[i] in actions:
            action_return, action_status = actions[p_text](
                conversation, parsed_text, i, is_or=is_or)
            return_statement += action_return

            # If operation fails, return error output to user
            if action_status == 0:
                break

            # This is a bit ugly but basically if an or occurs
            # we hold onto the or until we need to flip it off, i.e. filtering
            # happens
            if is_or is True and actions[p_text] == 'filter':
                is_or = False

        if p_text == 'or':
            is_or = True

    # Store 1-turn parsing
    conversation.store_last_parse(parsed_string)

    while return_statement.endswith("<br>"):
        return_statement = return_statement[:-len("<br>")]

    return return_statement


def run_action_new(conversation: Conversation,
                   question_id: str,
                   instance_id: int,
                   feature_id: int = None,
                   build_temp_dataset: bool = True,
                   instance_type_naming: str = "instance") -> str:
    """
    Runs the action selected by an ID instead of text parsing and updates the conversation object.

    conversation: Conversation, Conversation Object
    question_id: int, id of the question as defined in question_bank.csv
    instance_id: int, id of the instance that should be explained. Needed for local explanations
    feature_id: int, id of the feature name the question is about (if specified)
    build_temp_dataset: bool = True If building tmp_dataset is needed.
    """
    if build_temp_dataset:
        conversation.build_temp_dataset()

    # Create parse text as filter works with it
    parse_text = f"filter id {instance_id}".split(" ")
    _ = filter_operation(conversation, parse_text, 0)
    # Get tmp dataset to perform explanation on
    data = conversation.temp_dataset.contents['X']
    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    if feature_id is not None and feature_id != "":
        feature_name = data.columns[feature_id]
    parse_op = f"ID {instance_id}"
    model_prediction_probas, _ = predict_likelihood(conversation, as_text=False)
    current_prediction_str = conversation.get_class_name_from_label(np.argmax(model_prediction_probas))
    opposite_class = conversation.get_class_name_from_label(np.argmin(model_prediction_probas))
    current_prediction_id = conversation.temp_dataset.contents['y'][instance_id]
    template_manager = conversation.get_var('experiment_helper').contents.template_manager

    # First, check if follow-up question (id 0)
    if question_id == "followUp":
        # Get last parse to determine the last question
        last_question = conversation.get_last_question()
        q_id = last_question[0]
        f_id = last_question[1]
        if q_id is not None:
            question_id = q_id
            feature_id = f_id
        else:  # if follow up but no last method, set to 1
            question_id = 1

    # store the question ID and handle the question
    conversation.store_last_question(question_id, feature_id)

    if question_id == "notXaiMethod":  # Not XAI Method
        # First, check if first message
        if conversation.get_last_question()[0] is None:
            question_id = "greeting"  # switch to greeting
        else:
            return "I'm here to help you to understand the prediction of the Machine Learning " \
                   "model by showing you <br> " \
                   "<ul>" \
                   "<li>the <b>most</b> or <b>least</b> important attributes,</li>" \
                   "<li>the <b>strength and influence</b> of each attribute,</li>" \
                   "<li>which attribute changes would <b>switch the prediction</b> of the model,</li>" \
                   "<li>which attributes <b>guarantee this prediction</b>,</li>" \
                   "<li>the <b>distribution</b> of a single feature,</li>" \
                   "<li>or if the prediction changes by <b>altering a specific feature</b>.</li>" \
                   "</ul>"

    if question_id == "greeting":
        return "Hello, I am an assistant to help you understand the prediction of the Machine Learning model. You can " \
               "ask me about <br>" \
               "<ul>" \
               "<li>the <b>most</b> or <b>least</b> important attributes,</li>" \
               "<li>the <b>strength and influence</b> of each attribute,</li>" \
               "<li>which attribute changes would <b>switch the prediction</b> of the model,</li>" \
               "<li>which attributes <b>guarantee this prediction</b>,</li>" \
               "<li>the <b>distribution</b> of a single feature,</li>" \
               "<li>or if the prediction changes by <b>altering a specific feature</b>.</li>" \
               "</ul>"

    if question_id == "whyExplanation":
        return "To understand why the model made the prediction, I can tell you about" \
               "<ul>" \
               "<li>the <b>most</b> or <b>least</b> important attributes,</li>" \
               "<li>the <b>strength and influence</b> of each attribute,</li>" \
               "<li>which attribute changes would <b>switch the prediction</b> of the model,</li>" \
               "<li>which attributes </b>guarantee this prediction</b>,</li>" \
               "<li>how often a <b>single feature appears</b> or is spread out,</li>" \
               "<li>or if the prediction changes by </b>altering a specific feature</b>.</li>" \
               "</ul> <br> What would you like to know?"

    if question_id == "counterfactualAnyChange":
        # How should this instance change to get a different prediction?
        explanation, _ = explain_cfe(conversation, data, parse_op, regen)
        explanation = f"Here are possible scenarios that would change the prediction to <b>{opposite_class}</b>:<br> <br>" + \
                      explanation + "<br>There might be other possible changes. These are examples."
        return explanation
    if question_id == "counterfactualSpecificFeatureChange":
        # How should this attribute change to get a different prediction?
        top_features_dict, _ = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=False)
        explanation = explain_cfe_by_given_features(conversation, data, [feature_name], top_features_dict)
        return explanation
    if question_id == "anchor":
        # What attributes must be present or absent to guarantee this prediction?
        explanation, success = explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen,
                                                                                   template_manager)
        if success:
            result_text = f"Keeping these conditions: <br>"
            result_text = result_text + explanation + "<br>the prediction will most likely stay the same."
            return result_text
        else:
            return "I'm sorry, I couldn't find a group of attributes that guarantees the current prediction."
    if question_id == "featureStatistics":
        explanation = explain_feature_statistic(conversation, template_manager, feature_name, as_plot=True)
        return explanation
    if question_id == "top3Features":
        # 23;Which are the most important attributes for the outcome of the instance?
        parse_op = "top 3"
        explanation, _ = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=True,
                                                           template_manager=template_manager)
        answer = f"Here are the 3 <b>most</b> important attributes for predicting <b>{current_prediction_str}</b>: <br><br>"
        return answer + explanation[0]

    if question_id == "mostImportantFeature":
        # 23;Which are the most important attributes for the outcome of the instance?
        parse_op = "top 3"
        explanation, _ = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=False,
                                                           template_manager=template_manager)
        # Get first item in explanation dict
        most_important_feature_name = list(explanation[0].items())[0][0]
        return most_important_feature_name

    if question_id == "shapAllFeatures":
        explanation = explain_feature_importances_as_plot(conversation, data, parse_op, regen, current_prediction_str,
                                                          current_prediction_id)
        return explanation
    if question_id == "ceterisParibus":
        explanation = explain_ceteris_paribus(conversation, data, feature_name, instance_type_naming, opposite_class,
                                              as_text=True)
        if opposite_class not in explanation and not explanation.startswith("No"):
            explanation = explanation + opposite_class + "."
        return explanation
    if question_id == "least3Features":
        # 27;What features are used the least for prediction of the current instance?; What attributes are used the least for prediction of the instance?
        parse_op = "least 3"
        answer = f"Here are the <b>least</b> important attributes for predicting <b>{current_prediction_str}</b>: <br><br>"
        explanation, _ = explain_local_feature_importances(conversation, data, parse_op, regen, as_text=True,
                                                           template_manager=template_manager)
        return answer + explanation[0]
    else:
        return f"This is a mocked answer to your question with id {question_id}."


def compute_explanation_report(conversation,
                               instance_id: int,
                               build_temp_dataset: bool = True,
                               instance_type_naming: str = "instance",
                               feature_display_name_mapping=None):
    """
    Runs explanation methods on the current conversation and returns a static report.
    """
    if build_temp_dataset:
        conversation.build_temp_dataset()
    parse_text = f"filter id {instance_id}".split(" ")
    _ = filter_operation(conversation, parse_text, 0)
    data = conversation.temp_dataset.contents['X']
    regen = conversation.temp_dataset.contents['ids_to_regenerate']
    parse_op = f"ID {instance_id}"
    model_prediction_probas, _ = predict_likelihood(conversation, as_text=False)
    current_prediction_str = conversation.get_class_name_from_label(np.argmax(model_prediction_probas))
    current_prediction_id = conversation.temp_dataset.contents['y'][instance_id]

    model_prediction_str = conversation.get_class_name_from_label(np.argmax(model_prediction_probas))
    opposite_class = conversation.get_class_name_from_label(np.argmin(model_prediction_probas))
    template_manager = conversation.get_var("experiment_helper").contents.template_manager

    # Get already sorted feature importances
    feature_importances = explain_feature_importances_as_plot(conversation, data, parse_op, regen,
                                                              current_prediction_str,
                                                              current_prediction_id)

    cfe_string, desired_class = explain_cfe(conversation, data, parse_op, regen)
    counterfactual_strings = cfe_string + " <br>There are other possible changes. These are just examples."

    anchors_string, success = explain_anchor_changeable_attributes_without_effect(conversation, data, parse_op, regen,
                                                                                  template_manager)
    if success:
        result_text = f"Keeping these conditions: <br>"
        result_text = result_text + anchors_string + "<br>the prediction will most likely stay the same."
        anchors_string = result_text
    else:
        anchors_string = "There is no group of attributes that guarantees the current prediction."

    feature_statistics = explain_feature_statistic(conversation, template_manager, as_plot=False)
    # map feature names to display names
    if feature_display_name_mapping is not None:
        feature_statistics = {feature_display_name_mapping.get(key): value for key, value in
                              feature_statistics.items()}

    # get ceteris paribus for all features
    ceteris_paribus_sentences = []
    for feature in data.columns:
        ceteris_paribus = explain_ceteris_paribus(conversation, data, feature, instance_type_naming, opposite_class,
                                                  as_text=True)
        ceteris_paribus_sentences.append(ceteris_paribus)

    return {
        "model_prediction": model_prediction_str,
        "instance_type": instance_type_naming,
        "feature_importance": feature_importances,
        "opposite_class": opposite_class,
        "counterfactuals": counterfactual_strings,
        "anchors": anchors_string,
        "feature_statistics": feature_statistics,
        "ceteris_paribus": ceteris_paribus_sentences
    }
