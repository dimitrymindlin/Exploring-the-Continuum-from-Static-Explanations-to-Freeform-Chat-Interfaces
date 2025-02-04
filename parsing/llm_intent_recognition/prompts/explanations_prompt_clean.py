"""
Explanation intent prompt is the prompt to select an XAI or dialogue intent.
"""


def get_system_prompt_condensed_with_history(feature_names=""):
    return f"""<<Context>>:\n
    The user was presented with datapoint from a machine learning dataset with various features. The model predicted a class. Based on the 
        user's question about the prediction, follow the methods checklist to determine the best method.
        The possible feature names that the user might ask about are: {feature_names}\n
        
    <<Methods>>:\n
        - Greeting:
            - Examples: "Hey, how are you?", "Hello!", "Good morning."
            - JSON: method: "greeting", feature: None
        - What can you do?:
            - Examples: "What can you do?", "What explanations can you provide?", "How can you help me?"
            - JSON: method: "notXaiMethod", feature: None
        - Not stand alone, Short feature question without asking for a feature value change, feature value, or distribution:
            - Examples: "And what about age?", "income?", "Education level as well?"
            - JSON: method: "followUp", feature: "age" (or relevant feature)
        - Unspecific 'why' question:
            - Examples: "Why this prediction?", "What led to this result?"
            - JSON: method: "whyExplanation", feature: None
        - Not Xai Method and is rather a general or clarification question not related to model prediction?
            - Examples: "What does it mean?", "Can you clarify this term?", "How was the data collected?", "What is the accuracy of the model?", "What are the ethical implications of this model?"
            - JSON: method: "notXaiMethod", feature: None
        - Feature-specific or general XAI question:
            - Feature-specific:
                - Impact of changing a feature:
                    - Examples: "What if marital status was different?", "What if hours per week increased?", "What if older?"
                    - JSON: method: "ceterisParibus", feature: "marital status" (or relevant feature)
                - Feature statistics:
                    - Examples: "What are the typical values of 'age'?", "Can you show the statistics for this feature?"
                    - JSON: method: "featureStatistics", feature: "age" (or relevant feature)
            - General:
                - Impact of all features:
                    - Examples: "What is the strength of each feature?", "How much does each feature contribute?"
                    - JSON: method: "shapAllFeatures", feature: None
                - Top three features:
                    - Examples: "Which features had the greatest impact?", "What are the top factors influencing this result?"
                    - JSON: method: "top3Features", feature: None
                - Least three features:
                    - Examples: "Which features had the least impact?", "What are the least important factors?"
                    - JSON: method: "least3Features", feature: None
                - Class changes without specifying a feature:
                    - Examples: "Why is it not class [other class]?", "What changes would lead to a different prediction?"
                    - JSON: method: "counterfactualAnyChange", feature: "hours per week" (or relevant feature)
                - Anchoring conditions:
                    - Examples: "What factors guarantee this prediction?", "Which features must stay the same?"
                    - JSON: method: "anchor", feature: None

        <<Task>>:\n
        Decide which method fits best by reasoning over every possible method. 
        Return a single JSON with the following keys:
        
        <<Format Instructions>>:\n
        \n{{format_instructions}}
        
        <<Previous user questions and mapped methods>>:\n
        \n{{chat_history}}
        """


def simple_user_question_prompt_json_response():
    return f"""
    <<User Question>>:
    \n{{input}}
    \n
    <<Json Response>>:
    """


def openai_system_explanations_prompt(feature_names):
    return "system", get_system_prompt_condensed_with_history(feature_names)


def openai_user_prompt():
    return "user", simple_user_question_prompt_json_response()
