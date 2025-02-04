"""The main script that controls conversation logic.

This file contains the core logic for facilitating conversations. It orchestrates the necessary
routines for setting up conversations, controlling the state of the conversation, and running
the functions to get the responses to user inputs.
"""
import difflib
import pickle
from random import seed as py_random_seed
import secrets
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch

from flask import Flask
import gin

from create_experiment_data.diverse_instances_selection import DiverseInstances
from explain.action import run_action, run_action_new, compute_explanation_report
from explain.conversation import Conversation
from explain.decoder import Decoder
from explain.dialogue_manager.manager import DialogueManager
from explain.explainers.anchor_explainer import TabularAnchor
from explain.explainers.ceteris_paribus import CeterisParibus
from explain.explainers.feature_statistics_explainer import FeatureStatisticsExplainer
from explain.explanation import MegaExplainer
from explain.explainers.dice_explainer import TabularDice
from explain.parser import Parser, get_parse_tree
from explain.prompts import Prompts
from explain.utils import read_and_format_data
from create_experiment_data.experiment_helper import ExperimentHelper
from create_experiment_data.test_instances import TestInstances
from parsing.llm_intent_recognition.openai_pipeline import \
    LLMSinglePromptWithMemoryAndSystemMessage

# from explain.write_to_log import log_dialogue_input


app = Flask(__name__)


@gin.configurable
def load_sklearn_model(filepath):
    """Loads a sklearn model."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


@gin.configurable
class ExplainBot:
    """The ExplainBot Class."""

    def __init__(self,
                 study_group: str,
                 model_file_path: str,
                 dataset_file_path: str,
                 background_dataset_file_path: str,
                 dataset_index_column: int,
                 target_variable_name: str,
                 categorical_features: List[str],
                 ordinary_features: List[str],
                 numerical_features: List[str],
                 remove_underscores: bool,
                 name: str,
                 parsing_model_name: str = "ucinlp/diabetes-t5-small",
                 seed: int = 0,
                 prompt_metric: str = "cosine",
                 prompt_ordering: str = "ascending",
                 t5_config: str = None,
                 use_guided_decoding: bool = True,
                 feature_definitions: dict = None,
                 skip_prompts: bool = False,
                 instance_type_naming="instance",
                 use_intent_recognition: bool = False,
                 use_active_dialogue_manager: bool = False):
        """The init routine.

        Arguments:
            study_condition: The study condition for the experiment
            model_file_path: The filepath of the **user provided** model to explain. This model
                             should end with .pkl and support sklearn style functions like
                             .predict(...) and .predict_proba(...)
            dataset_file_path: The path to the dataset used in the conversation. Users will understand
                               the model's predictions on this dataset.
            background_dataset_file_path: The path to the dataset used for the 'background' data
                                          in the explanations.
            dataset_index_column: The index column in the data. This is used when calling
                                  pd.read_csv(..., index_col=dataset_index_column)
            target_variable_name: The name of the column in the dataset corresponding to the target,
                                  i.e., 'y'
            categorical_features: The names of the categorical features in the data. If None, they
                                  will be guessed.
            ordinary_features: The names of the ordinal features in the data.
            numerical_features: The names of the numeric features in the data. If None, they will
                                be guessed.
            remove_underscores: Whether to remove underscores in the feature names. This might help
                                performance a bit.
            name: The dataset name
            parsing_model_name: The name of the parsing model. See decoder.py for more details about
                                the allowed models.
            seed: The seed
            prompt_metric: The metric used to compute the nearest neighbor prompts. The supported options
                           are cosine, euclidean, and random
            prompt_ordering:
            t5_config: The path to the configuration file for t5 models, if using one of these.
            skip_prompts: Whether to skip prompt generation. This is mostly useful for running fine-tuned
                          models where generating prompts is not necessary.
            categorical_mapping_path: Path to json mapping for each col that assigns a categorical var to an int.
            use_intent_recognition: None or name of the intent recognition model to use.
        """

        # Set seeds
        np.random.seed(seed)
        py_random_seed(seed)
        torch.manual_seed(seed)

        self.bot_name = name
        self.study_group = study_group

        # Variables for experiment
        self.data_instances = []
        self.train_instance_counter = 0
        self.test_instance_counter = 0
        self.user_prediction_dict = {}
        self.current_instance = None
        self.current_instance_type = "train"  # Or test
        self.use_intent_recognition = use_intent_recognition
        self.categorical_features = categorical_features
        self.ordinary_features = ordinary_features
        self.instance_type_naming = instance_type_naming
        self.numerical_features = numerical_features
        self.use_active_dialogue_manager = use_active_dialogue_manager

        # Set up the conversation object
        self.conversation = Conversation(eval_file_path=dataset_file_path,
                                         feature_definitions=feature_definitions)

        # Load the model into the conversation
        self.load_model(model_file_path)

        # Load the dataset into the conversation
        self.load_dataset(dataset_file_path,
                          dataset_index_column,
                          target_variable_name,
                          categorical_features,
                          numerical_features,
                          remove_underscores,
                          store_to_conversation=True,
                          skip_prompts=skip_prompts)

        background_dataset, background_y_values = self.load_dataset(background_dataset_file_path,
                                                                    dataset_index_column,
                                                                    target_variable_name,
                                                                    categorical_features,
                                                                    numerical_features,
                                                                    remove_underscores,
                                                                    store_to_conversation=False)

        # Load Experiment Helper
        helper = ExperimentHelper(self.conversation, categorical_features)
        self.conversation.add_var('experiment_helper', helper, 'experiment_helper')

        if use_intent_recognition == "t5":
            # Prompt settings
            self.prompt_metric = prompt_metric
            self.prompt_ordering = prompt_ordering
            self.use_guided_decoding = use_guided_decoding

            # A variable used to help file uploads
            self.manual_var_filename = None
            self.decoding_model_name = parsing_model_name

            # Initialize completion + parsing modules
            app.logger.info(f"Loading parsing model {parsing_model_name}...")
            self.decoder = Decoder(parsing_model_name,
                                   t5_config,
                                   use_guided_decoding=self.use_guided_decoding,
                                   dataset_name=name)

            # Initialize parser + prompts as None
            # These are done when the dataset is loaded
            self.prompts = None
            self.parser = None
        elif self.use_intent_recognition == "promptBasedLLM":
            self.intent_recognition_model = LLMSinglePromptWithMemoryAndSystemMessage(helper.feature_ordering)
        else:
            self.intent_recognition_model = None

        # Load the explanations
        self.load_explanations(background_dataset=background_dataset, background_y_values=background_y_values)

        ## Initialize Dialogue Manager
        self.dialogue_manager = DialogueManager(intent_recognition=self.intent_recognition_model,
                                                template_manager=helper.template_manager,
                                                active=self.use_active_dialogue_manager)

    def init_loaded_var(self, name: bytes):
        """Inits a var from manual load."""
        self.manual_var_filename = name.decode("utf-8")

    def get_next_instance_triple(self, instance_type, return_probability=False):
        """
        Returns the next instance in the data_instances list if possible.
        param instance_type: type of instance to return, can be train, test or final_test
        """
        experiment_helper = self.conversation.get_var('experiment_helper').contents
        self.current_instance, counter, self.current_instance_type = experiment_helper.get_next_instance(
            instance_type=instance_type,
            return_probability=return_probability)
        return self.current_instance, counter

    def get_feature_importances_for_current_instance(self):
        mega_explainer = self.conversation.get_var('mega_explainer').contents
        instance = self.current_instance[1]
        data = pd.DataFrame(instance, index=[self.current_instance[0]])
        feature_importance_dict = mega_explainer.get_feature_importances(data, [], False)[0]
        # Turn display names into feature names
        feature_importance_dict = {self.get_feature_display_name_dict().get(k, k): v for k, v in
                                   feature_importance_dict.items()}
        # Extract the feature importances for the current instance from outer dict with current class as key
        current_prediction_class = self.current_instance[4]
        feature_importance_dict = feature_importance_dict[current_prediction_class]
        return feature_importance_dict

    def reset_dialogue_manager(self):
        """
        Resets the dialogue manager to the initial state and resets the feature importances for the current instance.
        """
        current_feature_importances = self.get_feature_importances_for_current_instance()
        self.dialogue_manager.reset_state()
        self.dialogue_manager.feature_importances = current_feature_importances

    def get_suggested_method(self):
        return self.dialogue_manager.get_suggested_explanations()

    def get_study_group(self):
        """Returns the study group."""
        return self.study_group

    def get_feature_display_name_dict(self):
        template_manager = self.conversation.get_var("experiment_helper").contents.template_manager
        return template_manager.feature_display_names.feature_name_to_display_name

    def get_current_prediction(self):
        """
        Returns the current prediction.
        """
        # Can be either [2], then argmax, or [3] then its a string
        if isinstance(self.current_instance[2], np.ndarray):
            current_prediction = np.argmax(self.current_instance[2])
            prediction_string = self.conversation.class_names[current_prediction]
        else:
            prediction_string = self.current_instance[3]  # This is the prediction string
        return prediction_string

    def set_user_prediction(self, user_prediction):
        true_label = self.get_current_prediction()
        current_id = self.current_instance[0]
        reversed_dict = {value: key for key, value in self.conversation.class_names.items()}
        true_label_as_int = reversed_dict[true_label]
        try:
            user_prediction_as_int = reversed_dict[user_prediction]
        except KeyError:
            user_prediction_as_int = int(1000)
            # for "I don't know" option
        print(f"User prediction: {user_prediction_as_int}, True label: {true_label_as_int}")
        # Make 2d dict with self.current_instance_type as first key and current_id as second key
        if self.current_instance_type not in self.user_prediction_dict:
            self.user_prediction_dict[self.current_instance_type] = {}

        self.user_prediction_dict[self.current_instance_type][current_id] = (user_prediction_as_int, true_label_as_int)

    def get_user_correctness(self, train=False):
        # Check self.user_prediction_dict for correctness
        correct_counter = 0
        total_counter = 0
        # Get correct prediction dict
        if train:
            predictions_dict = self.user_prediction_dict["train"]
        else:
            predictions_dict = self.user_prediction_dict["test"]
        # Calculate correctness
        for instance_id, (user_prediction, true_label) in predictions_dict.items():
            if user_prediction == true_label:
                correct_counter += 1
            total_counter += 1
        correctness_string = f"{correct_counter} out of {total_counter}"
        return correctness_string

    def load_explanations(self, background_dataset, background_y_values):
        """Loads the explanations.

        If set in gin, this routine will cache the explanations.

        Arguments:
            background_dataset: The background dataset to compute the explanations with.
        """
        app.logger.info("Loading explanations into conversation...")

        # This may need to change as we add different types of models
        pred_f = self.conversation.get_var('model_prob_predict').contents
        model = self.conversation.get_var('model').contents
        data = self.conversation.get_var('dataset').contents['X']
        categorical_f = self.conversation.get_var('dataset').contents['cat']
        numeric_f = self.conversation.get_var('dataset').contents['numeric']
        test_data_y = self.conversation.get_var('dataset').contents['y']

        exp_helper = self.conversation.get_var("experiment_helper").contents

        # Load lime tabular explanations
        mega_explainer = MegaExplainer(prediction_fn=pred_f,
                                       data=background_dataset,
                                       cat_features=categorical_f,
                                       class_names=self.conversation.class_names,
                                       categorical_mapping=exp_helper.categorical_mapping)

        # Load diverse instances (explanations)
        app.logger.info("...loading DiverseInstances...")
        diverse_instances_explainer = DiverseInstances(
            lime_explainer=mega_explainer.mega_explainer.explanation_methods['lime_0.75'])
        diverse_instance_ids = diverse_instances_explainer.get_instance_ids_to_show(data=data,
                                                                                    model=model,
                                                                                    y_values=test_data_y,
                                                                                    submodular_pick=False)
        # Make new list of dicts {id: instance_dict} where instance_dict is a dict with column names as key and values as values.
        diverse_instances = [{"id": i, "values": data.loc[i].to_dict()} for i in diverse_instance_ids]

        # Load mega explainer explanations
        mega_explainer.get_explanations(ids=diverse_instance_ids, data=data)
        message = (f"...loaded {len(mega_explainer.cache)} mega explainer "
                   "explanations from cache!")
        app.logger.info(message)
        # Load dice explanations
        tabular_dice = TabularDice(model=model,
                                   data=data,
                                   num_features=numeric_f,
                                   class_names=self.conversation.class_names,
                                   background_dataset=background_dataset,
                                   features_to_vary=exp_helper.actionable_features)
        tabular_dice.get_explanations(ids=diverse_instance_ids,
                                      data=data)
        message = (f"...loaded {len(tabular_dice.cache)} dice tabular "
                   "explanations from cache!")
        app.logger.info(message)

        # Load anchor explanations
        tabular_anchor = TabularAnchor(model=model,
                                       data=data,
                                       class_names=self.conversation.class_names,
                                       feature_names=list(data.columns),
                                       categorical_mapping=exp_helper.categorical_mapping)
        tabular_anchor.get_explanations(ids=diverse_instance_ids,
                                        data=data)

        # Load Ceteris Paribus Explanations
        ceteris_paribus_explainer = CeterisParibus(model=model,
                                                   background_data=background_dataset,
                                                   ys=background_y_values,
                                                   class_names=self.conversation.class_names,
                                                   feature_names=list(data.columns),
                                                   categorical_mapping=exp_helper.categorical_mapping,
                                                   ordinal_features=self.ordinary_features)
        ceteris_paribus_explainer.get_explanations(ids=diverse_instance_ids,
                                                   data=data)

        # Load FeatureStatisticsExplainer with background data
        feature_statistics_explainer = FeatureStatisticsExplainer(background_dataset,
                                                                  background_y_values,
                                                                  self.numerical_features,
                                                                  feature_names=list(background_dataset.columns),
                                                                  rounding_precision=self.conversation.rounding_precision,
                                                                  categorical_mapping=exp_helper.categorical_mapping,
                                                                  feature_units=exp_helper.template_manager.feature_units_mapping)
        self.conversation.add_var('feature_statistics_explainer', feature_statistics_explainer, 'explanation')

        # Add all the explanations to the conversation
        self.conversation.add_var('diverse_instances', diverse_instances, 'diverse_instances')
        self.conversation.add_var('mega_explainer', mega_explainer, 'explanation')
        self.conversation.add_var('tabular_dice', tabular_dice, 'explanation')
        self.conversation.add_var('tabular_anchor', tabular_anchor, 'explanation')
        self.conversation.add_var('ceteris_paribus', ceteris_paribus_explainer, 'explanation')

        # Load test instances
        test_instance_explainer = TestInstances(data, model, mega_explainer,
                                                self.conversation.get_var("experiment_helper").contents,
                                                diverse_instance_ids=diverse_instance_ids,
                                                actionable_features=exp_helper.actionable_features)
        test_instances = test_instance_explainer.get_test_instances()
        self.conversation.add_var('test_instances', test_instances, 'test_instances')

    def load_model(self, filepath: str):
        """Loads a model.

        This routine loads a model into the conversation
        from a specified file path. The model will be saved as a variable
        names 'model' in the conversation, overwriting an existing model.

        The routine determines the type of model from the file extension.
        Scikit learn models should be saved as .pkl's and torch as .pt.

        Arguments:
            filepath: the filepath of the model.
        Returns:
            success: whether the model was saved successfully.
        """
        app.logger.info(f"Loading inference model at path {filepath}...")
        if filepath.endswith('.pkl'):
            model = load_sklearn_model(filepath)
            self.conversation.add_var('model', model, 'model')
            self.conversation.add_var('model_prob_predict',
                                      model.predict_proba,
                                      'prediction_function')
        else:
            # No other types of models implemented yet
            message = (f"Models with file extension {filepath} are not supported."
                       " You must provide a model stored in a .pkl that can be loaded"
                       f" and called like an sklearn model.")
            raise NameError(message)
        app.logger.info("...done")
        return 'success'

    def load_dataset(self,
                     filepath: str,
                     index_col: int,
                     target_var_name: str,
                     cat_features: list[str],
                     num_features: list[str],
                     remove_underscores: bool,
                     store_to_conversation: bool,
                     skip_prompts: bool = False):
        """Loads a dataset, creating parser and prompts.

        This routine loads a dataset. From this dataset, the parser
        is created, using the feature names, feature values to create
        the grammar used by the parser. It also generates prompts for
        this particular dataset, to be used when determine outputs
        from the model.

        Arguments:
            filepath: The filepath of the dataset.
            index_col: The index column in the dataset
            target_var_name: The target column in the data, i.e., 'y' for instance
            cat_features: The categorical features in the data
            num_features: The numeric features in the data
            remove_underscores: Whether to remove underscores from feature names
            store_to_conversation: Whether to store the dataset to the conversation.
            skip_prompts: whether to skip prompt generation.
        Returns:
            success: Returns success if completed and store_to_conversation is set to true. Otherwise,
                     returns the dataset.
        """
        app.logger.info(f"Loading dataset at path {filepath}...")

        # Read the dataset and get categorical and numerical features
        dataset, y_values, categorical, numeric = read_and_format_data(filepath,
                                                                       index_col,
                                                                       target_var_name,
                                                                       cat_features,
                                                                       num_features,
                                                                       remove_underscores)

        if store_to_conversation:

            # Store the dataset
            self.conversation.add_dataset(dataset, y_values, categorical, numeric)

            if self.use_intent_recognition == "t5":
                # Set up the parser
                self.parser = Parser(cat_features=categorical,
                                     num_features=numeric,
                                     dataset=dataset,
                                     target=list(y_values))

                # Generate the available prompts
                # make sure to add the "incorrect" temporary feature
                # so we generate prompts for this
                self.prompts = Prompts(cat_features=categorical,
                                       num_features=numeric,
                                       target=np.unique(list(y_values)),
                                       feature_value_dict=self.parser.features,
                                       class_names=self.conversation.class_names,
                                       skip_creating_prompts=skip_prompts)
                app.logger.info("..done")

            return "success"
        else:
            return dataset, y_values

    def set_num_prompts(self, num_prompts):
        """Updates the number of prompts to a new number"""
        self.prompts.set_num_prompts(num_prompts)

    @staticmethod
    def gen_almost_surely_unique_id(n_bytes: int = 30):
        """To uniquely identify each input, we generate a random 30 byte hex string."""
        return secrets.token_hex(n_bytes)

    @staticmethod
    def log(logging_input: dict):
        """Performs the system logging."""
        assert isinstance(logging_input, dict), "Logging input must be dict"
        assert "time" not in logging_input, "Time field will be added to logging input"
        # log_dialogue_input(logging_input)

    @staticmethod
    def build_logging_info(bot_name: str,
                           username: str,
                           response_id: str,
                           system_input: str,
                           parsed_text: str,
                           system_response: str):
        """Builds the logging dictionary."""
        return {
            'bot_name': bot_name,
            'username': username,
            'id': response_id,
            'system_input': system_input,
            'parsed_text': parsed_text,
            'system_response': system_response
        }

    def compute_parse_text(self, text: str, error_analysis: bool = False):
        """Computes the parsed text from the user text input.

        Arguments:
            error_analysis: Whether to do an error analysis step, where we compute if the
                            chosen prompts include all the
            text: The text the user provides to the system
        Returns:
            parse_tree: The parse tree from the formal grammar decoded from the user input.
            parse_text: The decoded text in the formal grammar decoded from the user input
                        (Note, this is just the tree in a string representation).
        """
        nn_prompts = None
        if error_analysis:
            grammar, prompted_text, nn_prompts = self.compute_grammar(text, error_analysis=error_analysis)
        else:
            grammar, prompted_text = self.compute_grammar(text, error_analysis=error_analysis)
        app.logger.info("About to decode")
        # Do guided-decoding to get the decoded text
        api_response = self.decoder.complete(
            prompted_text, grammar=grammar)
        decoded_text = api_response['generation']

        app.logger.info(f'Decoded text {decoded_text}')

        # Compute the parse tree from the decoded text
        # NOTE: currently, we're using only the decoded text and not the full
        # tree. If we need to support more complicated parses, we can change this.
        parse_tree, parsed_text = get_parse_tree(decoded_text)
        if error_analysis:
            return parse_tree, parsed_text, nn_prompts
        else:
            return parse_tree, parsed_text,

    def get_feature_tooltips(self):
        """
        Returns the feature tooltips for the current dataset.
        """
        return self.conversation.get_var("experiment_helper").contents.template_manager.feature_tooltip_mapping

    def get_feature_units(self):
        """
        Returns the feature units for the current dataset.
        """
        return self.conversation.get_var("experiment_helper").contents.template_manager.feature_units_mapping

    def get_questions_attributes_featureNames(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns the questions and attributes and feature names for the current dataset.
        """
        try:
            # Read the question bank CSV file
            question_pd = pd.read_csv(self.conversation.question_bank_path, delimiter=";")

            # Replace "instance" in all 'paraphrased' entries with instance_type_naming
            question_pd["paraphrased"] = question_pd["paraphrased"].str.replace("instance", self.instance_type_naming)

            # Create answer dictionary with general and feature questions
            answer_dict = {
                "general_questions": question_pd[question_pd["question_type"] == "general"]
                                     .loc[:, ['q_id', 'paraphrased']]
                .rename(columns={'paraphrased': 'question'})
                .to_dict(orient='records'),

                "feature_questions": question_pd[question_pd["question_type"] == "feature"]
                                     .loc[:, ['q_id', 'paraphrased']]
                .rename(columns={'paraphrased': 'question'})
                .to_dict(orient='records')
            }

            return answer_dict

        except FileNotFoundError:
            raise Exception(f"File not found: {self.conversation.question_bank_path}")
        except pd.errors.EmptyDataError:
            raise Exception("The question bank CSV file is empty or invalid.")

    def get_feature_names(self):
        """
        Returns the feature names for the current dataset.
        """
        return self.conversation.get_var("experiment_helper").contents.get_feature_names()

    def compute_parse_text_t5(self, text: str):
        """Computes the parsed text for the input using a t5 model.

        This supposes the user has finetuned a t5 model on their particular task and there isn't
        a need to do few shot
        """
        grammar, prompted_text = self.compute_grammar(text)
        decoded_text = self.decoder.complete(text, grammar)
        app.logger.info(f"t5 decoded text {decoded_text}")
        parse_tree, parse_text = get_parse_tree(decoded_text[0])
        return parse_tree, parse_text

    def compute_grammar(self, text, error_analysis: bool = False):
        """Computes the grammar from the text.

        Arguments:
            text: the input text
            error_analysis: whether to compute extra information used for error analyses
        Returns:
            grammar: the grammar generated for the input text
            prompted_text: the prompts computed for the input text
            nn_prompts: the knn prompts, without extra information that's added for the full
                        prompted_text provided to prompt based models.
        """
        nn_prompts = None
        app.logger.info("getting prompts")
        # Compute KNN prompts
        if error_analysis:
            prompted_text, adhoc, nn_prompts = self.prompts.get_prompts(text,
                                                                        self.prompt_metric,
                                                                        self.prompt_ordering,
                                                                        error_analysis=error_analysis)
        else:
            prompted_text, adhoc = self.prompts.get_prompts(text,
                                                            self.prompt_metric,
                                                            self.prompt_ordering,
                                                            error_analysis=error_analysis)
        app.logger.info("getting grammar")
        # Compute the formal grammar, making modifications for the current input
        grammar = self.parser.get_grammar(
            adhoc_grammar_updates=adhoc)

        if error_analysis:
            return grammar, prompted_text, nn_prompts
        else:
            return grammar, prompted_text

    def update_state_ttm(self, text: str, user_session_conversation: Conversation):
        """The main conversation driver.

        The function controls state updates of the conversation. It accepts the
        user input and ultimately returns the updates to the conversation.

        Arguments:
            text: The input from the user to the conversation.
            user_session_conversation: The conversation sessions for the current user.
        Returns:
            output: The response to the user input.
        """

        if any([text is None, self.prompts is None, self.parser is None]):
            return ''

        app.logger.info(f'USER INPUT: {text}')

        # Parse user input into text abiding by formal grammar
        if "t5" not in self.decoding_model_name:
            parse_tree, parsed_text = self.compute_parse_text(text)
        else:
            parse_tree, parsed_text = self.compute_parse_text_t5(text)

        # Run the action in the conversation corresponding to the formal grammar
        returned_item = run_action(
            user_session_conversation, parse_tree, parsed_text)

        username = user_session_conversation.username

        response_id = self.gen_almost_surely_unique_id()
        logging_info = self.build_logging_info(self.bot_name,
                                               username,
                                               response_id,
                                               text,
                                               parsed_text,
                                               returned_item)
        self.log(logging_info)
        # Concatenate final response, parse, and conversation representation
        # This is done so that we can split both the parse and final
        # response, then present all the data
        final_result = returned_item + f"<>{response_id}"

        return final_result

    def get_explanation_report(self):
        """Returns the explanation report."""
        instance_id = self.current_instance[0]
        report = compute_explanation_report(self.conversation, instance_id,
                                            instance_type_naming=self.instance_type_naming,
                                            feature_display_name_mapping=self.get_feature_display_name_dict())
        return report

    def update_state_experiment(self,
                                question_id: int = None,
                                feature_id: int = None) -> tuple[str, int, Optional[int]]:
        """The main experiment driver.

                The function controls state updates of the conversation. It accepts the
                user input as question_id and feature_id and returns the updates to the conversation.

                Arguments:
                    question_id: The question id from the user.
                    feature_id: The feature id that the question is about.
                Returns:
                    output: The response to the user input.
                """

        instance_id = self.current_instance[0]
        question_id, feature_name, reasoning = self.dialogue_manager.update_state(None, question_id, feature_id)

        if question_id is None:
            return '', None, None, reasoning

        if feature_id is not None and feature_id != "":
            feature_id = int(feature_id)

        app.logger.info(f'USER INPUT: q_id:{question_id}, f_id:{feature_id}')
        # Convert feature_id to int if not None
        returned_item = run_action_new(self.conversation,
                                       question_id,
                                       instance_id,
                                       feature_id,
                                       instance_type_naming=self.instance_type_naming)

        # self.log(logging_info) # Logging dict currently off.
        # Concatenate final response, parse, and conversation representation
        # This is done so that we can split both the parse and final
        # response, then present all the data
        # final_result = returned_item + f"<>{response_id}"
        final_result = returned_item
        return final_result, question_id, feature_id, reasoning

    async def update_state_from_nl(self, user_input):
        # 1. Get the question_id and feature_name from the user input
        feature_id = None

        # Get the question_id and feature_name from the user input
        question_id, feature_name, reasoning = self.dialogue_manager.update_state(user_input)
        if feature_name != "" and feature_name is not None:
            feature_list = [col.lower() for col in self.conversation.stored_vars['dataset'].contents['X'].columns]
            # remove whitespace between words
            feature_name = feature_name.replace(" ", "")
            try:
                feature_id = feature_list.index(feature_name.lower())
            except ValueError:
                # Get closest match
                closest_matches = difflib.get_close_matches(feature_name, feature_list, n=1, cutoff=0.5)
                if closest_matches:
                    feature_id = feature_list.index(closest_matches[0])
                else:
                    feature_id = None
                    # Optionally handle the case where no close match is found
                    print(f"No close match found for feature name: {feature_name}")

        # 2. Update the state
        return self.update_state_experiment(question_id, feature_id)
