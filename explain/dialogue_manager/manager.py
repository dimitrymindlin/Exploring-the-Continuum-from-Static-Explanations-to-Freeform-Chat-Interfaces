from typing import Dict

from explain.dialogue_manager.dialogue_policy import DialoguePolicy


class DialogueManager:
    def __init__(self, intent_recognition, template_manager, active=True):
        # If the dialogue manager is active and suggests explanations or passive and just answers questions
        self.active_mode = active
        self.intent_recognition_model = intent_recognition
        self.template_manager = template_manager
        if self.active_mode:
            self.dialogue_policy = DialoguePolicy()
        else:
            self.dialogue_policy = None
        self.user_questions = 0
        self.interacted_explanations = set()
        self.feature_importances: Dict = None
        # Count position of next feature to explain
        self.ceteris_paribus_features_explained = set()
        self.feature_statistics_explained = set()

    def reset_state(self):
        """
        Reset the state of the dialogue manager for new interactions
        """
        if self.active_mode:
            self.dialogue_policy.reset_state()
        self.user_questions = 0
        self.interacted_explanations = set()
        self.most_important_attribute = None
        self.ceteris_paribus_features_explained.clear()
        self.feature_statistics_explained.clear()

    def get_next_feature(self, method_name):
        """
        Get the next feature for the method cp or feature statistics by considering the features that were already explained.
        Take the next feature from the feature importances that was not explained yet for the given method.
        """
        assert method_name in ["ceterisParibus", "featureStatistics"]
        already_explained = None
        if method_name == "ceterisParibus":
            already_explained = self.ceteris_paribus_features_explained
        elif method_name == "featureStatistics":
            already_explained = self.feature_statistics_explained
        for feature_id, importance in self.feature_importances.items():
            if feature_id not in already_explained:
                return feature_id
        return None

    def mark_as_explained(self, explanation, feature):
        # get feature name from feature id
        if isinstance(feature, int):
            feature_name = self.template_manager.conversation.get_var("dataset").contents["X"].columns[feature]
        else:
            feature_name = feature
        if explanation == "ceterisParibus":
            self.ceteris_paribus_features_explained.add(feature_name)
        elif explanation == "featureStatistics":
            self.feature_statistics_explained.add(feature_name)

    def update_state(self, user_input, question_id=None, feature_id=None):
        """
        Update the state of the dialogue manager based on the user input. If the question_id is not None, the user
        clicked on a question and the dialogue manager can update the state machine directly. If the question_id is None,
        the user input needs NLU to determine the intent and the feature_id, then the state machine is updated.
        :param user_input: The user input
        :param question_id: The id of the question the user clicked on
        :param feature_id: The id of the feature the user clicked on
        :return: The id of the suggested explanation, the id of the feature the user clicked on, and the suggested followups
        """
        self.user_questions += 1
        if question_id is not None:
            # Direct mapping, update state machine
            self.interacted_explanations.add(question_id)
            if self.active_mode:
                self.dialogue_policy.model.trigger(question_id)
                self.mark_as_explained(question_id, feature_id)
            single_reasoning_string = "Clicked on question"
            return question_id, feature_id, single_reasoning_string

        # If question_id is None, the user input needs NLU
        intent_classification = None
        method_name = None
        feature_name = None
        single_reasoning_string = ""

        # (step 1) Get user Intent
        if self.active_mode:
            intent_classification, method_name, feature_name, reasoning_1 = self.intent_recognition_model.interpret_user_answer(
                self.get_suggested_explanations(),
                user_input)
            single_reasoning_string = "Step1:" + reasoning_1

        # (step 2) If the intent is not recognized or the dialogue manager is not active, predict the explanation method
        if not self.active_mode or intent_classification == "other":
            method_name, feature_name, reasoning_2 = self.intent_recognition_model.predict_explanation_method(
                user_input)
            single_reasoning_string += "Step2:" + reasoning_2

        # Update the state machine
        if self.active_mode:
            self.dialogue_policy.model.trigger(method_name)
            self.interacted_explanations.add(method_name)
            self.mark_as_explained(method_name, feature_name)
        return method_name, feature_name, single_reasoning_string

    def replace_most_important_attribute(self, suggested_followups):
        updated_followups = []
        for followup in suggested_followups:
            if "most important attribute" in followup['question']:
                method = followup['id']
                next_feature_to_explain = self.get_next_feature(method)
                if next_feature_to_explain is None:
                    continue  # Skip adding this followup to the updated list because all features were explained
                display_name = self.template_manager.get_feature_display_name_by_name(next_feature_to_explain)
                followup['question'] = followup['question'].replace("most important attribute", display_name)
                followup['feature'] = next_feature_to_explain
            updated_followups.append(followup)

        # Replace the original list with the updated list
        suggested_followups[:] = updated_followups
        return suggested_followups

    def get_suggested_explanations(self):
        suggested_followups = self.dialogue_policy.get_suggested_followups()
        self.replace_most_important_attribute(suggested_followups)
        return suggested_followups

    def print_transitions(self):
        if self.active_mode:
            self.dialogue_policy.to_mermaid()

    def get_proceeding_okay(self):
        """
        Checks if the user asked more than 2 questions. If yes, the user is okay with the explanation.
        If not, check which questions the user did not ask yet and return them.
        :return: Tuple of boolean and list of questions the user did not ask yet
        """
        if self.user_questions > 2:
            return True, None, ""
        else:
            if self.active_mode:
                not_asked_yet = self.dialogue_policy.get_not_asked_questions()
                not_asked_yet = self.replace_most_important_attribute(not_asked_yet)
                try:
                    not_asked_yet = not_asked_yet[:3]
                except (IndexError, TypeError):
                    not_asked_yet = []
                return False, not_asked_yet[
                              :3], "Already want to proceed? Maybe you have some more questions ... ?"
            else:
                return False, None, "Already want to proceed? You could ask about the most or least important attribute," \
                                    "the influences of features or about which changes would lead to a different prediction..."
