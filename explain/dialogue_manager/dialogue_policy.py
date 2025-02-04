# from transitions import Machine
from transitions import Machine


class Model:
    pass


class DialoguePolicy:
    # Define the states
    states = [
        'initial',
        'shownTop3',
        'shownLeast3',
        'shownShapAll',
        'shownCeterisParibus',
        'shownCounterfactualAnyChange',
        'shownFeatureStatistics',
        'shownAnchor',
        'shownNotXaiMethod',
        'shownFollowUp',
        'shownWhyExplanation',
        'shownGreeting',
    ]

    # Define the method names as variables
    SHOW_TOP3 = 'top3Features'
    SHOW_LEAST3 = 'least3Features'
    SHOW_SHAP_ALL = 'shapAllFeatures'
    SHOW_CETERIS_PARIBUS = 'ceterisParibus'
    SHOW_FEATURE_STATISTICS = 'featureStatistics'
    SHOW_COUNTERFACTUAL_ANY_CHANGE = 'counterfactualAnyChange'
    SHOW_ANCHOR = 'anchor'
    SHOW_NOT_XAI_METHOD = 'notXaiMethod'
    SHOW_FOLLOW_UP = 'followUp'
    SHOW_WHY_EXPLANATION = 'whyExplanation'
    SHOW_GREETING = 'greeting'

    # Define the transitions using the method name variables
    transitions = [
        {'trigger': SHOW_TOP3, 'source': '*', 'dest': 'shownTop3'},
        {'trigger': SHOW_LEAST3, 'source': '*', 'dest': 'shownLeast3'},
        {'trigger': SHOW_SHAP_ALL, 'source': '*', 'dest': 'shownShapAll'},
        {'trigger': SHOW_CETERIS_PARIBUS, 'source': '*', 'dest': 'shownCeterisParibus'},
        {'trigger': SHOW_FEATURE_STATISTICS, 'source': '*', 'dest': 'shownFeatureStatistics'},
        {'trigger': SHOW_COUNTERFACTUAL_ANY_CHANGE, 'source': '*', 'dest': 'shownCounterfactualAnyChange'},
        {'trigger': SHOW_ANCHOR, 'source': '*', 'dest': 'shownAnchor'},
        {'trigger': SHOW_NOT_XAI_METHOD, 'source': '*', 'dest': 'shownNotXaiMethod'},
        {'trigger': SHOW_FOLLOW_UP, 'source': '*', 'dest': 'shownFollowUp'},
        {'trigger': SHOW_WHY_EXPLANATION, 'source': '*', 'dest': 'shownWhyExplanation'},
        {'trigger': SHOW_GREETING, 'source': '*', 'dest': 'shownGreeting'},
    ]

    # Define the questions for each transition
    questions = {
        SHOW_TOP3: "Would you like to see the top 3 important attributes?",
        SHOW_LEAST3: "Would you like to see the least important attributes?",
        SHOW_SHAP_ALL: "Would you like to see the impact of all attributes?",
        SHOW_CETERIS_PARIBUS: "Would you like to see if changing the most important attribute can change the model prediction?",
        SHOW_FEATURE_STATISTICS: "Would you like to see the different values and distribution of the most important attribute?",
        SHOW_COUNTERFACTUAL_ANY_CHANGE: "Would you like to see possible changes that flip the model prediction?",
        SHOW_ANCHOR: "Would you like to see which group of attributes most definitely predicts the current outcome?"
    }

    followups = {
        'initial': [SHOW_TOP3, SHOW_ANCHOR, SHOW_COUNTERFACTUAL_ANY_CHANGE],
        'shownTop3': [SHOW_LEAST3, SHOW_SHAP_ALL, SHOW_CETERIS_PARIBUS],
        'shownLeast3': [SHOW_SHAP_ALL, SHOW_COUNTERFACTUAL_ANY_CHANGE],
        'shownShapAll': [SHOW_COUNTERFACTUAL_ANY_CHANGE, SHOW_CETERIS_PARIBUS, SHOW_FEATURE_STATISTICS],
        'shownCeterisParibus': [SHOW_CETERIS_PARIBUS, SHOW_FEATURE_STATISTICS],
        'shownFeatureStatistics': [SHOW_FEATURE_STATISTICS],
        'shownCounterfactualAnyChange': [SHOW_CETERIS_PARIBUS],
        'shownAnchor': [SHOW_COUNTERFACTUAL_ANY_CHANGE],
        'shownNotXaiMethod': [],  # Suggest Most Used XAI Methods
        'shownFollowUp': [SHOW_CETERIS_PARIBUS, SHOW_FEATURE_STATISTICS],  # Suggest Follow Up Questions
        'shownWhyExplanation': [SHOW_SHAP_ALL, SHOW_ANCHOR, SHOW_COUNTERFACTUAL_ANY_CHANGE],
        'shownGreeting': [],
    }

    def __init__(self):
        # Create the state machine
        self.model = Model()
        self.machine = Machine(model=self.model, states=DialoguePolicy.states, initial='initial')
        self.asked_questions = []

        # Add transitions dynamically
        for transition in DialoguePolicy.transitions:
            trigger = transition['trigger']
            dest = transition['dest']
            self.machine.add_transition(trigger, source='*', dest=dest, before=self.ask_question)

    def ask_question(self, *args, **kwargs):
        trigger = kwargs.get('trigger')
        if trigger:
            self.asked_questions.append(trigger)
            print(DialoguePolicy.questions[trigger])

    def get_suggested_followups(self):
        current_state = self.model.state
        return [{'id': trigger, 'question': DialoguePolicy.questions[trigger], 'feature': None} for trigger in
                DialoguePolicy.followups.get(current_state, [])]

    def get_last_explanation(self):
        try:
            return self.asked_questions[-1]
        except IndexError:
            return None

    def get_not_asked_questions(self, num_questions=2):
        """
        Get the next num_questions that were not asked yet, avoiding follow ups from the current state.
        If there are not enough not asked questions, get the rest from the follow ups.
        return: List of dictionaries with id, question and feature keys
        """
        followups = DialoguePolicy.followups.get(self.model.state, [])
        not_asked_questions = [q for q in DialoguePolicy.questions if
                               q not in self.asked_questions and q not in followups]

        # Return required not asked questions, filling the rest with followups if needed
        questions = not_asked_questions[:num_questions]
        if len(questions) < num_questions:
            questions.extend(followups[:num_questions - len(not_asked_questions)])

        return [{'id': q, 'question': DialoguePolicy.questions[q], 'feature': None} for q in questions]

    def reset_state(self):
        """
        Reset the state of the dialogue policy for new interactions
        """
        self.machine.set_state('initial')

    def to_mermaid(self, include_trigger=False):
        mermaid_def = "stateDiagram-v2\n"
        for state in self.states:
            state = state.replace('shown', '')
            mermaid_def += f"    {state}\n"
        for state, followup_triggers in self.followups.items():
            for trigger in followup_triggers:
                dest = next(t['dest'] for t in self.transitions if t['trigger'] == trigger)
                state = state.replace('shown', '')
                dest = dest.replace('shown', '')

                if include_trigger:
                    mermaid_def += f"    {state} --> {dest} : {trigger}\n"
                else:
                    mermaid_def += f"    {state} --> {dest}\n"
        print(mermaid_def)
