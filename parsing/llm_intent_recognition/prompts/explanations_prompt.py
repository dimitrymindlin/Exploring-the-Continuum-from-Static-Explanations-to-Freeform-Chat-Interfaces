"""
Explanation intent prompt is the prompt to select an XAI or dialogue intent.
"""

whyExplanation_template = """
whyExplanation: Explains possible explanations when the user asks a general why question.
    Best for questions like: "Why this prediction?", "What led to this result?", "Can you explain why the model chose this class?"
    Answer example: "To understand the prediction, I can tell you about the most important attributes, or 
    which changes would have led to a different prediction."
"""

greeting_template = """
greeting: Greets the user when they do not ask a specific question. Inform them about the types of questions you can answer.
    Best for questions like: "Hey, how are you?", "Hello!", "Good morning."
    Answer example: "Hello, I am here to help you understand the prediction of the Machine Learning model.
    You can ask about the most or least important attributes..."
"""

not_xai_method_template = """
notXaiMethod: Used for questions that cannot be answered with the previous methods, such as clarifications or 
    domain-related questions.
    Best for questions like: "What does it mean?", "Can you clarify this term?", "What is meant by 'X' in this context?"
    Answer example: "[Description of the questioned thing]".
"""

followUp_template = """
followUp: Not stand alone, Short feature question without asking for a change, value, or distribution.
    Best for questions like: "And what about age?", "How does income affect it?", "What if we consider education level?"
    Answer Example: "Here is the same explanation method for the new feature."
"""

anchor_template = """
anchor: Identifies the minimal set of conditions and feature values that ensure a specific prediction remains unchanged.
    Best for questions like: "What factors guarantee this prediction remains the same?", "Which features must stay the same for this result?", "How can we be sure this prediction won't change?"
    Answer Example: "If 'age' remains constant, the model's prediction will remain unchanged."
"""

shapeAllFeatures_template = """
shapAllFeatures: Identifies each feature's individual contribution to the overall prediction.
      Best for questions like: "What is the strength of each feature?", "How much does each feature contribute?", "Can you show the impact of all features?"
      Answer Example: "Here is the contribution of each feature..."
"""

top3Features_template = """
top3Features: Identifies the top three features that have the highest impact on the model's prediction.
      Best for questions like: "Which features had the greatest impact on this prediction?", "What are the top factors influencing this result?", "Most important features?"
      Answer Example: "The top 3 features are 'occupation', and 'hours per week' and 'age'."
"""

least3Features_template = """
least3Features: Identifies three features with the least impact on the model’s prediction.
    Best for questions like: "Which features had the least impact on this prediction?", "What are the least important factors?", "Can you show the features that matter the least?"
    Answer Example: "The least influential features were 'marital status', 'race', and 'sex'."
"""

counterfactualAnyChange_template = """
counterfactualAnyChange: Identifies close instances where the model's prediction changes, exploring feature alterations that lead to different predictions.
      Best for questions like: "Why is it not class [other class]?", "In which case would it be another class?", "What changes would lead to a different prediction?"
      Answer Example: "The prediction would switch from class A to class B if 'hours per week' increased by more than 10 hours."
"""

featureStatistics_template = """
featureStatistics: Identifies statistical summary of the features in a dataset, like minimum, maximum, mean, and standard deviation.
      Best for questions like: "What are the typical values and distributions of 'age' in my dataset?", "Can you show the statistics for this feature?", "What is the average value of this attribute?"
      Answer Example: "The mean age is 40 years with a standard deviation of 12.5."
"""

ceterisParibus_template = """
ceterisParibus: Examines the impact of changing one single feature to explore if the model's prediction changes.
      Best for questions like: "What would happen if marital status was different?", "What if hours per week increased?", "How would the prediction change if this feature value was altered?"
      Answer Example: "Changing 'marital status' from 'single' to 'married' would result in a prediction change."
"""


def get_template_with_full_descriptions(feature_names):
    return f"""The user was presented an instance with different features.
        The machine learning model predicted a class. Given the user question about the model prediction, 
        decide which method fits best to answer it. There are standalone methods that work without requiring a feature 
        specification: {general_questions} and some that are specific to a feature: {feature_specific_q}.
        Here are possible feature names that the user might ask about: {feature_names}.
        
    Here are definitions for the available methods:
    - {whyExplanation_template}
    - {greeting_template}
    - {not_xai_method_template}
    - {followUp_template}
    - {anchor_template}
    - {shapeAllFeatures_template}
    - {top3Features_template}
    - {least3Features_template}
    - {counterfactualAnyChange_template}
    - {featureStatistics_template}
    - {ceterisParibus_template}
""" + """
Decide which methods fits best and if its a feature specific one, also reply with the feature.
Immediately answer with the method and feature if applicable, without justification.

<<User Question>>
\n{input}

Respond with a python tuple containing the method and feature if applicable. feature can be None.
"""


def get_template_with_full_descr_step_by_step():
    return f"""
    The user was presented with an instance containing various features. The machine learning model predicted a class. 
    Given the user's question about the model's prediction, select the most suitable method to answer it. There are 
    standalone methods that work without requiring a specific feature: {general_questions}, and methods that are 
    specific to a feature: {feature_specific_q}.

    Here are definitions for the available methods:
    - {whyExplanation_template}
    - {greeting_template}
    - {not_xai_method_template}
    - {followUp_template}
    - {anchor_template}
    - {shapeAllFeatures_template}
    - {top3Features_template}
    - {least3Features_template}
    - {counterfactualAnyChange_template}
    - {featureStatistics_template}
    - {ceterisParibus_template}
""" + """
    Decide which method fits best and, if it’s a feature-specific one, also reply with the feature. First, describe your 
    thougth process and finally answer with the method and feature in a json format.
    
    << User question >>
    \n{input}
    
    << Answer >>
    Let's think about the right method step by step. First, let's consider if it's a follow up question or a 
    question that can be mapped to a specific method. Then, if its a general or feature-specific question.
    If its a feature-specific question, we need to identify the feature. Finally, respond with a python tuple containing the method and feature if applicable. 
    feature can be None.
    """


def get_template_wich_checklist(feature_names="are defined by the user."):
    return f"""The user was presented with an instance containing various features. The machine learning model predicted 
        a class. Given the user's question about the model's prediction, follow the checklist to determine the most suitable 
        method to answer it. Return ONLY a json containing the keys method and feature where feature can be None. Do not
         justify the choice of the method or feature, just provide the json. Possible feature names: {feature_names}

    1. First, check if it is a greeting:
        - Example questions: "Hey, how are you?", "Hello!", "Good morning."
        - JSON response: method_name: "greeting", feature: None

    2. If it is not a greeting, check if it is a very short question about a feature without specifying a change or value question (high or low, average...):
        - Example questions: "And what about age?", "income?", "Education level as well?", "And income?."
        - JSON response: method_name: "followUp", feature: "age" (or the relevant feature mentioned)

    3. If it is not a followUp question, check if it is a unspecific 'why did this happen' question. The user is interested
        in understanding the prediction but does not ask for a specific explanation:
        - Example questions: "Why this prediction?", "What led to this result?", "Can you explain why the model chose this class?"
        - JSON response: method_name: "whyExplanation", feature: None

    4. If it is not a a greeting, followUp or general whyExplanations, check if the question is a feature-specific or general xai question:
        - If feature-specific:
            - If asking for the impact of changing a specific feature:
                - Example questions: "What would happen if marital status was different?", "What if hours per week increased?", "How would the prediction change if this feature value was altered?"
                - JSON response: method_name: "ceterisParibus", feature: "marital status" (or the relevant feature mentioned)
            - If asking for feature statistics:
                - Example questions: "What are the typical values and distributions of 'age' in my dataset?", "Can you show the statistics for this feature?", "What is the average value of this attribute?"
                - JSON response: method_name: "featureStatistics", feature: "age" (or the relevant feature mentioned)
            - If asking for the anchor features:
                - Example questions: "What factors guarantee this prediction remains the same?", "Which features must stay the same for this result?", "How can we be sure this prediction won't change?"
                - JSON response: method_name: "anchor", feature: None
        - If general:
            - If asking for the impact of all features:
                - Example questions: "What is the strength of each feature?", "How much does each feature contribute?", "Can you show the impact of all features?"
                - JSON response: method_name: "shapAllFeatures", feature: None
            - If asking for the top three features:
                - Example questions: "Which features had the greatest impact on this prediction?", "What are the top factors influencing this result?", "Can you show the most important features?"
                - JSON response: method_name: "top3Features", feature: None
            - If asking for the least three features:
                - Example questions: "Which features had the least impact on this prediction?", "What are the least important factors?", "Can you show the features that matter the least?"
                - JSON response: method_name: "least3Features", feature: None
             If asking for class changes, without specifying a feature:
                - Example questions: "Why is it not class [other class]?", "In which case would it be another class?", "What changes would lead to a different prediction?"
                - JSON response: method_name: "counterfactualAnyChange", feature: "hours per week" (or the relevant feature mentioned)
                
    5. If it's not specific XAI question of the above, check if the question is not related to the model prediction but is a
        clarification question or dataset related question, i.e. it is not directed to the model prediction:
        - If it is not related to XAI, use `notXaiMethod`.
            - Example questions: "What does it mean?", "Can you clarify this term?", "What is meant by 'X' in this context?", "How was the data collected?", "What is the accuracy of the model?", "What are the ethical implications of this model?"
            - JSON response: method_name: "notXaiMethod", feature: None

    Decide which method fits best. Return a single json containing the keys method and feature where feature can be None.
    Do not justify the choice of the method or feature, just provide the json.
    
    <<User Question>>
    \n{{input}}
    """


def get_system_template_with_checklist():
    return f"""The user was presented with an instance containing various features. The machine learning model predicted 
        a class. Given the user's question about the model's prediction, follow the checklist to determine the most suitable 
        method to answer it. Return ONLY a json containing the keys method and feature where feature can be None. Do not
         justify the choice of the method or feature, just provide the json.

    1. First, check if it is a greeting:
        - Example questions: "Hey, how are you?", "Hello!", "Good morning."
        - JSON response: method_name: "greeting", feature: None

    2. If it is not a greeting, check if it is a very short question about a feature without specifying a change or asking for the value:
        - Example questions: "And what about age?", "income?", "Education level as well?", "And income?."
        - JSON response: method_name: "followUp", feature: "age" (or the relevant feature mentioned)

    3. If it is not a followUp question, check if it is a unspecific 'why did this happen' question. The user is interested
        in understanding the prediction but does not ask for a specific explanation:
        - Example questions: "Why this prediction?", "What led to this result?", "Can you explain why the model chose this class?"
        - JSON response: method_name: "whyExplanation", feature: None

    4. If it is not a a greeting, followUp or general whyExplanations, check if the question is a feature-specific or general xai question:
        - If feature-specific:
            - If asking for the impact of changing a specific feature:
                - Example questions: "What would happen if marital status was different?", "What if hours per week increased?", "How would the prediction change if this feature value was altered?"
                - JSON response: method_name: "ceterisParibus", feature: "marital status" (or the relevant feature mentioned)
            - If asking for feature statistics:
                - Example questions: "What are the typical values and distributions of 'age' in my dataset?", "Can you show the statistics for this feature?", "What is the average value of this attribute?"
                - JSON response: method_name: "featureStatistics", feature: "age" (or the relevant feature mentioned)
            - If asking for the anchor features:
                - Example questions: "What factors guarantee this prediction remains the same?", "Which features must stay the same for this result?", "How can we be sure this prediction won't change?"
                - JSON response: method_name: "anchor", feature: None
        - If general:
            - If asking for the impact of all features:
                - Example questions: "What is the strength of each feature?", "How much does each feature contribute?", "Can you show the impact of all features?"
                - JSON response: method_name: "shapAllFeatures", feature: None
            - If asking for the top three features:
                - Example questions: "Which features had the greatest impact on this prediction?", "What are the top factors influencing this result?", "Can you show the most important features?"
                - JSON response: method_name: "top3Features", feature: None
            - If asking for the least three features:
                - Example questions: "Which features had the least impact on this prediction?", "What are the least important factors?", "Can you show the features that matter the least?"
                - JSON response: method_name: "least3Features", feature: None
             If asking for class changes, without specifying a feature:
                - Example questions: "Why is it not class [other class]?", "In which case would it be another class?", "What changes would lead to a different prediction?"
                - JSON response: method_name: "counterfactualAnyChange", feature: "hours per week" (or the relevant feature mentioned)

    5. If it's not specific XAI question of the above, check if the question is not related to the model prediction but is a
        clarification question or dataset related question, i.e. it is not directed to the model prediction:
        - If it is not related to XAI, use `notXaiMethod`.
            - Example questions: "What does it mean?", "Can you clarify this term?", "What is meant by 'X' in this context?", "How was the data collected?", "What is the accuracy of the model?", "What are the ethical implications of this model?"
            - JSON response: method_name: "notXaiMethod", feature: None

    Decide which method fits best. Return a single json containing the keys method and feature where feature can be None.
    Do not justify the choice of the method or feature, just provide the json.
    """


def get_system_prompt_condensed(feature_names=[]):
    return f"""The user was presented with an instance with various features. The model predicted a class. Based on the 
        user's question about the prediction, follow the checklist to determine the best method. Return ONLY a JSON with 
        'method' and 'feature'. 'feature' can be None. Do not justify the choice. Possible feature names are: {feature_names}

        1. Greeting:
            - Examples: "Hey, how are you?", "Hello!", "Good morning."
            - JSON: method: "greeting", feature: None
        2. Not stand alone, Short feature question without asking for a change, value, or distribution:
            - Examples: "And what about age?", "income?", "Education level as well?"
            - JSON: method: "followUp", feature: "age" (or relevant feature)
        3. Unspecific 'why' question:
            - Examples: "Why this prediction?", "What led to this result?"
            - JSON: method: "whyExplanation", feature: None
        4. Not Xai Method and is rather a general or clarification question not related to model prediction?
            - Examples: "What does it mean?", "Can you clarify this term?", "How was the data collected?", "What is the accuracy of the model?", "What are the ethical implications of this model?"
            - JSON: method: "notXaiMethod", feature: None
        5. Feature-specific or general XAI question:
            - Feature-specific:
                - Impact of changing a feature:
                    - Examples: "What if marital status was different?", "What if hours per week increased?"
                    - JSON: method: "ceterisParibus", feature: "marital status" (or relevant feature)
                - Feature statistics:
                    - Examples: "What are the typical values of 'age'?", "Can you show the statistics for this feature?"
                    - JSON: method: "featureStatistics", feature: "age" (or relevant feature)
                - Anchoring conditions:
                    - Examples: "What factors guarantee this prediction?", "Which features must stay the same?"
                    - JSON: method: "anchor", feature: None
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
                    
        Task: Decide which method fits best. Return a single JSON.
        
        {{format_instructions}}
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
        Decide which method fits best by reasoning over every possible method. If you choose a feature specific method,
        the feature cannot be None. If the user did not provide a specific feature, the method should be a fitting 
        general one. Return a single JSON with the following keys:
        
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


def simple_user_question_prompt():
    return f"""
    <<User Question>>
    \n{{question}}
    \n
    <<Response>>
    """


def openai_system_explanations_prompt(feature_names):
    return "system", get_system_prompt_condensed_with_history(feature_names)


def openai_user_prompt():
    return "user", simple_user_question_prompt_json_response()


def get_template_with_checklist_condensed(feature_names):
    return f"""
    You are given a user's question about a model's prediction. Use the checklist to determine the suitable method.

    1. Check if it's a greeting:
        - Examples: "Hey, how are you?", "Hello!", "Good morning."
        - Response: method_name: "greeting", feature: None

    2. If not a greeting, check if it's a short feature question:
        - Examples: "And what about age?", "Income?", "Education level?"
        - Response: method_name: "followUp", feature: "age" (or relevant feature)

    3. If not a followUp, check if it's a general 'why' question:
        - Examples: "Why this prediction?", "What led to this result?"
        - Response: method_name: "whyExplanation", feature: None

    4. If not above, check if it's a specific or general XAI question:
        - Specific feature:
            - Changing feature impact:
                - Examples: "What if marital status was different?", "How would hours per week change prediction?"
                - Response: method_name: "ceterisParibus", feature: "marital status" (or relevant feature)
            - Feature statistics:
                - Examples: "Typical values of 'age'?", "Show feature statistics?"
                - Response: method_name: "featureStatistics", feature: "age" (or relevant feature)
            - Anchor features:
                - Examples: "What guarantees this prediction?", "Which features ensure this result?"
                - Response: method_name: "anchor", feature: "age" (or relevant feature)
        - General:
            - Impact of all features:
                - Examples: "Feature strength?", "Show impact of all features?"
                - Response: method_name: "shapAllFeatures", feature: None
            - Top three features:
                - Examples: "Top impacting features?", "Most important factors?"
                - Response: method_name: "top3Features", feature: None
            - Least three features:
                - Examples: "Least impacting features?", "Least important factors?"
                - Response: method_name: "least3Features", feature: None
            - Class changes without specifying feature:
                - Examples: "Why not class [other class]?", "What changes lead to different prediction?"
                - Response: method_name: "counterfactualAnyChange", feature: None

    5. If not related to model prediction but a clarification or dataset question:
        - Examples: "What does it mean?", "Clarify term?", "Data collection method?", "Model accuracy?", "Ethical implications?"
        - Response: method_name: "notXaiMethod", feature: None

    Possible features: {feature_names}

    user
    <<User Question>>
    \n{{input}}

    assistant
    Respond with a python tuple (method, feature). Feature can be None.

    <<Answer>>
        """


def get_template_wich_checklist_and_memory(feature_names):
    return f"""The following is an intent recognition sequence in a conversation between an AI and a human user and 
    about machine learning model decisions. The AI is helpful and understands what the users intent is, given his question.
     The user was presented with an instance containing various features. The machine learning model predicted a class. 
     Given the user's question, the AI follows the checklist to determine the most suitable method to answer it. 

    1. First, check if it is a greeting:
        - Example questions: "Hey, how are you?", "Hello!", "Good morning."
        - JSON response: method_name: "greeting", feature: None

    2. If it is not a greeting, check if it is a very short question about a feature without specifying a change or statistics:
        - Example questions: "And what about feature1?", "feature2?", "feature1 as well?", "And feature3?."
        - JSON response: method_name: "followUp", feature: "feature1" (or the relevant feature mentioned)

    3. If it is not a followUp question, check if it is a unspecific 'why did this happen' question. The user is interested
        in understanding the prediction but does not ask for a specific explanation:
        - Example questions: "Why this prediction?", "What led to this result?", "Can you explain why the model chose this class?"
        - JSON response: method_name: "whyExplanation", feature: None

    4. If it is not a a greeting, followUp or general whyExplanations, check if the question is a feature-specific or general xai question:
        - If feature-specific:
            - If asking for the impact of changing a specific feature:
                - Example questions: "What would happen if marital status was different?", "What if hours per week increased?", "How would the prediction change if this feature value was altered?"
                - JSON response: method_name: "ceterisParibus", feature: "marital status" (or the relevant feature mentioned)
            - If asking for feature statistics:
                - Example questions: "What are the typical values and distributions of 'feature1' in my dataset?", "Can you show the statistics for this feature?", "What is the average value of this attribute?"
                - JSON response: method_name: "featureStatistics", feature: "feature1" (or the relevant feature mentioned)
            - If asking for the anchor features:
                - Example questions: "What factors guarantee this prediction remains the same?", "Which features must stay the same for this result?", "How can we be sure this prediction won't change?"
                - JSON response: method_name: "anchor", feature: None
        - If general:
            - If asking for the impact of all features:
                - Example questions: "What is the strength of each feature?", "How much does each feature contribute?", "Can you show the impact of all features?"
                - JSON response: method_name: "shapAllFeatures", feature: None
            - If asking for the top three features:
                - Example questions: "Which features had the greatest impact on this prediction?", "What are the top factors influencing this result?", "Can you show the most important features?"
                - JSON response: method_name: "top3Features", feature: None
            - If asking for the least three features:
                - Example questions: "Which features had the least impact on this prediction?", "What are the least important factors?", "Can you show the features that matter the least?"
                - JSON response: method_name: "least3Features", feature: None
             If asking for class changes, without specifying a feature:
                - Example questions: "Why is it not class [other class]?", "In which case would it be another class?", "What changes would lead to a different prediction?"
                - JSON response: method_name: "counterfactualAnyChange", feature: "hours per week" (or the relevant feature mentioned)

    5. If it's not specific XAI question of the above, check if the question is not related to the model prediction but is a
        clarification question or dataset related question, i.e. it is not directed to the model prediction:
        - If it is not related to XAI, use `notXaiMethod`.
            - Example questions: "What does it mean?", "Can you clarify this term?", "What is meant by 'X' in this context?", "How was the data collected?", "What is the accuracy of the model?", "What are the ethical implications of this model?"
            - JSON response: method_name: "notXaiMethod", feature: None

    The AI decide which method fits best. It immediately answer with the method and feature if applicable, without justification.
    Possible feature that the user might ask about: {feature_names}.

    <<Previous user questions and mapped methods>>:
    \n{{chat_history}}

    <<Current User Question>>
    \n{{question}}

    The AI responds with a python tuple containing the method and feature if applicable. feature can be None.
    
    <<AI Response>>
    """


def get_template_with_checklist_plus_agreement(feature_names):
    return f"""The following is an intent recognition sequence in a conversation between an AI and a human user and 
    about machine learning model decisions. The AI is helpful and understands what the users intent is, given his question.
     The user was presented with an instance containing various features. The machine learning model predicted a class. 
     Given the user's question, the AI thinks step by step and follows the checklist to determine the most suitable method to answer it. 
     
    First, check if the user agreed to a suggested method:
        The suggested methods are shown before the user question.
        - Example questions: "Yes, show me the top 3 features.", "Okay, would like to see the feature statistics.", "Yes.", "Good Idea."
        - JSON response: method_name: agreement, feature: method name if applicable.
        
    Then, check if the user simply disagreed or is interested in seeing other explanation possibilities:
        - Example questions: "No, I would like to see the impact of all features.", "I disagree, show me the anchor features.", "I don't think so, show me the least 3 features.", "I would like to see something else."
        - JSON response: method_name: rejection, feature: None

    If not, check if it is a greeting:
        - Example questions: "Hey, how are you?", "Hello!", "Good morning."
        - JSON response: method_name: "greeting", feature: None

    If it is not a greeting, check if it is a very short question about a feature without specifying a change or statistics:
        - Example questions: "And what about feature1?", "feature2?", "feature1 as well?", "And feature3?."
        - JSON response: method_name: "followUp", feature: "feature1" (or the relevant feature mentioned)

    If it is not a followUp question, check if it is a unspecific 'why did this happen' question. The user is interested
        in understanding the prediction but does not ask for a specific explanation:
        - Example questions: "Why this prediction?", "What led to this result?", "Can you explain why the model chose this class?"
        - JSON response: method_name: "whyExplanation", feature: None

    If it is not a a greeting, followUp or general whyExplanations, check if the question is a feature-specific or general xai question:
        - If feature-specific:
            - If asking for the impact of changing a specific feature:
                - Example questions: "What would happen if marital status was different?", "What if hours per week increased?", "How would the prediction change if this feature value was altered?"
                - JSON response: method_name: "ceterisParibus", feature: "marital status" (or the relevant feature mentioned)
            - If asking for feature statistics:
                - Example questions: "What are the typical values and distributions of 'feature1' in my dataset?", "Can you show the statistics for this feature?", "What is the average value of this attribute?"
                - JSON response: method_name: "featureStatistics", feature: "feature1" (or the relevant feature mentioned)
            - If asking for the anchor features:
                - Example questions: "What factors guarantee this prediction remains the same?", "Which features must stay the same for this result?", "How can we be sure this prediction won't change?"
                - JSON response: method_name: "anchor", feature: None
        - If general:
            - If asking for the impact of all features:
                - Example questions: "What is the strength of each feature?", "How much does each feature contribute?", "Can you show the impact of all features?"
                - JSON response: method_name: "shapAllFeatures", feature: None
            - If asking for the top three features:
                - Example questions: "Which features had the greatest impact on this prediction?", "What are the top factors influencing this result?", "Can you show the most important features?"
                - JSON response: method_name: "top3Features", feature: None
            - If asking for the least three features:
                - Example questions: "Which features had the least impact on this prediction?", "What are the least important factors?", "Can you show the features that matter the least?"
                - JSON response: method_name: "least3Features", feature: None
             If asking for class changes, without specifying a feature:
                - Example questions: "Why is it not class [other class]?", "In which case would it be another class?", "What changes would lead to a different prediction?"
                - JSON response: method_name: "counterfactualAnyChange", feature: "hours per week" (or the relevant feature mentioned)

    If it's not specific XAI question of the above, check if the question is not related to the model prediction but is a
        clarification question or dataset related question, i.e. it is not directed to the model prediction:
        - If it is not related to XAI, use `notXaiMethod`.
            - Example questions: "What does it mean?", "Can you clarify this term?", "What is meant by 'X' in this context?", "How was the data collected?", "What is the accuracy of the model?", "What are the ethical implications of this model?"
            - JSON response: method_name: "notXaiMethod", feature: None

    The AI decide which method fits best. It thinks step by step and returns the final answer as a json.
    Possible feature that the user might ask about: {feature_names}.
    
    """


def get_xai_template_with_descriptions():
    return f"""The user was presented an instance with different features.
        The machine learning model predicted a class. Given the user question about the model prediction, 
        decide which method fits best to answer it. There are standalone methods that work without requiring a feature 
        specification: {general_questions} and some that are specific to a feature: {feature_specific_q}.""" + """
    
    REMEMBER: "method" MUST be one of the candidate method names specified below OR it can be None if the input is not well suited for any of the candidate prompts.
    REMEMBER: "feature_name" can be None if the question is not related to a specific feature.
    
    << CANDIDATE METHODS >>
    - anchor: Anchor explanations identify the minimal set of conditions and feature values that, when held constant, 
    ensure a specific prediction result does not change, irrespective of alterations to other features. 
    Best for questions like "What factors guarantee this prediction remains the same?"
    Answer example: "If feature1 and feature2 remain constant, the model's prediction will remain unchanged."
    - shapAllFeatures: uses SHAP values to measure and visualize all feature's individual contribution to the 
    overall prediction, considering the marginal effect of each feature across all possible combinations. 
    This comprehensive approach provides a complete overview by showing the impact of all features of the instance.
    Best for questions like: "What is the strength of each feature?"
    Answer example: "Here is a visualization, showing the contribution of each feature..."
    - top3Features: This method focuses on identifying and visualizing the contributions of the top three 
    features that have the highest impact on the model's prediction, according to SHAP values. It simplifies the 
    explanation by concentrating on the most influential variables.
    Best for questions like: "Which features had the greatest impact on this prediction?"
    Answer example: 'The top influences on the prediction were feature1, feature2, and feature3.'"
    - least3Features: This method concentrates on the three features with the least impact on the model’s
    prediction, according to SHAP values. It provides insights into the features that have minimal influence on
    the outcome, which can be critical for understanding the robustness of the model or for identifying potential
    areas of model simplification.
    Best for questions like: "Which features had the least impact on this prediction?"
    Answer example: 'The least influential features were feature1, feature2, and feature3, each contributing minimally 
    to the overall prediction according to their SHAP values.'
    - ceterisParibus: This method focuses on looking for the impact of changes in a feature requested by the 
    user. It is useful to investigate what would happen if some feature was different. i.e higher or lower.
     Best for questions like: "What would happen if feature1 was different?"
     Answer example: "Changing feature1 from value1 to value2 would result in the prediction changing from class1 to class2."
    - counterfactualAnyChange: Provide possible feature alterations to understand scenarios under which the model's 
    prediction is changed. This method is suited for exploring changes in the features of the current instance that 
    would lead to a different prediction, thereby clarifying why the current instance didn't classify as another category. 
    Best for questions like: 'Why is it not class2?', or 'In which case would be other class?'
    Answer example: 'The prediction would switch from class1 to class2 if feature1 increased by value1, or feature2 was 
    above value2 or ...'
    - featureStatistics: Provides a statistical summary or visual representation of the features in a dataset.
    It calculates the mean and standard deviation for numerical features, offering a clear quantitative 
    overview of data spread and central tendency, giving intuition about whether a certain current value is 
    particularly high, low or average. 
    Best for questions like: "What are the typical values and distributions of feature1 in my dataset?"
    Answer example: "The mean of feature1 is value1 with a standard deviation of value2."

\n{format_instructions}

REMEMBER: Do not justify the choice of the method or feature, just provide the method and feature.

<< User Question >>
\n{input}

<< Answer >>
"""


general_questions = [
    "anchor",
    "shapAllFeatures",
    "top3Features",
    "least3Features",
    "counterfactualAnyChange"]
feature_specific_q = [
    "ceterisParibus",
    "featureStatistics",
]
extra_dialogue_intents = [
    "followUp",
    "whyExplanation",
    "notXaiMethod",
    "greeting"
]
possible_categories = general_questions + feature_specific_q + extra_dialogue_intents
possible_features = [
    'Age',
    'EducationLevel',
    'MaritalStatus',
    'Occupation',
    'WeeklyWorkingHours',
    'WorkLifeBalance',
    'InvestmentOutcome']
question_to_id_mapping = {
    "top3Features": 23,
    "anchor": 11,
    "shapAllFeatures": 24,
    "least3Features": 27,
    "ceterisParibus": 25,
    "featureStatistics": 13,
    "counterfactualAnyChange": 7,
    "followUp": 0,
    "whyExplanation": 1,
    "notXaiMethod": 100,
    "greeting": 99,
    "None": -1,
    "agreement": 101,
    "rejection": 102
}

"""response_schemas = [
    ResponseSchema(name="method_name", description="name of the method to answer the user question."),
    ResponseSchema(name="feature", description="Feature that the user mentioned, can be None."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()"""

ROUTING_TASK_PROMPT = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. Copy the input into the next_inputs field.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""
