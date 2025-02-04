import re


class FeatureDisplayNames:
    def __init__(self, conversation, display_names_mapping):
        self.conversation = conversation
        self.feature_name_to_display_name = display_names_mapping

    def get_by_id(self, feature_id):
        """
        Function to get feature display name by feature id
        :param feature_id: feature id
        :return: feature display name
        """
        return self.feature_name_to_display_name.get(feature_id)

    def get_by_name(self, feature_name):
        return self.feature_name_to_display_name[feature_name]
