"""
User response prompt is the prompt to check if the user response can be interpreted as a response to the suggested
method or is otherwise a different question.
"""


def initial_routing_prompt_template(feature_names):
    return f"""
    <<Context>>: \n The user is asked if they would like to see a suggested XAI (Explainable AI) method.\n 

    <<Suggested Methods>>:
    \n{{explanation_suggestions}}\n
    
    <<Task>>:
    Given the user’s response, classify it into “agreement”, if the user agrees to any of the suggested methods, 
    or “other” if the user asks something else.\n
    The user might agree to a feature specific method and reference one of the features: {feature_names}.\n
    
    <<Expected Output>>:
    Return the reasoning and classification in JSON format with the following keys:\n
    - “reasoning”: The reasoning behind the classification of “agreement“ or “other“. The reasoning on whether the
    user agrees to a suggestion or wants to know something else should be explained and the method name should be 
    mentioned if the classification is “agreement”.
    - “classification”: Decide on ne of [“agreement”, “other”] based on the reasoning
	- “method”: If classification is “agreement”, the method ID from the suggested methods should be returned.
	- “feature”: One of the feature names if mentioned by the user, otherwise null
    """


def openai_system_prompt_initial_routing(feature_names):
    return "system", initial_routing_prompt_template(feature_names)


def openai_user_prompt_initial_routing():
    return "user", f"""
    <<User Response>>:
    \n{{user_response}}
    """
