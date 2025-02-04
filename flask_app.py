"""The app main."""
import json
import logging
from logging.config import dictConfig
import os
import traceback
from flask import Flask, jsonify
from flask import render_template, request, Blueprint
import gin
from flask_cors import CORS

from explain.logic import ExplainBot
from explain.sample_prompts_by_action import sample_prompt_for_action


# gunicorn doesn't have command line flags, using a gin file to pass command line args
@gin.configurable
class GlobalArgs:
    def __init__(self, config, baseurl):
        self.config = config
        self.baseurl = baseurl


# Parse gin global config
gin.parse_config_file("global_config.gin")

# Get args
args = GlobalArgs()

bp = Blueprint('host', __name__, template_folder='templates')

CORS(bp)

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse application level configs
gin.parse_config_file(args.config)

# Setup the explainbot dict to run multiple bots
bot_dict = {}

TTM_Bot = ExplainBot("")


@bp.route('/')
def home():
    """Load the explanation interface."""
    # Setup the explainbot
    app.logger.info("Loaded Login")
    objective = TTM_Bot.conversation.describe.get_dataset_objective()
    return render_template("index.html", currentUserId="user", datasetObjective=objective)


@bp.route('/init', methods=['GET'])
def init_experiment():
    """Load the explanation interface."""
    user_id = request.args.get("user_id")
    study_group = request.args.get("study_group")
    if user_id is None or "":
        user_id = "TEST"
    if study_group is None or "":
        study_group = "interactive"
    BOT = ExplainBot(study_group)
    bot_dict[user_id] = BOT
    app.logger.info("Loaded Login and created bot")

    # Feature tooltip and units
    feature_tooltip = bot_dict[user_id].get_feature_tooltips()
    feature_units = bot_dict[user_id].get_feature_units()
    questions = bot_dict[user_id].get_questions_attributes_featureNames()
    ordered_feature_names = bot_dict[user_id].get_feature_names()
    user_experiment_prediction_choices = bot_dict[user_id].conversation.class_names
    user_study_task_description = bot_dict[user_id].conversation.describe.get_user_study_objective()
    result = {
        "feature_tooltips": feature_tooltip,
        "feature_units": feature_units,
        'questions': questions,
        'feature_names': ordered_feature_names,
        'prediction_choices': user_experiment_prediction_choices,
        'user_study_task_description': user_study_task_description
    }
    return result


def get_datapoint(user_id, datapoint_type, return_probability=False):
    """
    Get a datapoint from the dataset based on the datapoint type.
    """
    if user_id is None:
        user_id = "TEST"
    current_instance_with_units, instance_counter = bot_dict[user_id].get_next_instance_triple(datapoint_type,
                                                                                               return_probability=return_probability)
    (instance_id, instance_dict, probas, ml_label, _) = current_instance_with_units
    instance_dict["id"] = str(instance_id)
    if return_probability:
        instance_dict["probabilities"] = probas
    instance_dict["ml_prediction"] = ml_label
    return instance_dict


@bp.route('/get_train_datapoint', methods=['GET'])
def get_train_datapoint():
    """
    Get a new datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    user_study_group = bot_dict[user_id].get_study_group()
    result_dict = get_datapoint(user_id, "train")
    bot_dict[user_id].reset_dialogue_manager()

    if user_study_group == "interactive":
        prompt = f"""
            The model predicts that the current {bot_dict[user_id].instance_type_naming} is <b>{result_dict["ml_prediction"]}</b>. <br>
            If you have questions about the prediction, select questions from the right and I will answer them.
            """
    else:  # chat
        prompt = f"""
            The model predicts that the current {bot_dict[user_id].instance_type_naming} is <b>{result_dict["ml_prediction"]}</b>. <br>
            If you have questions about the prediction, <b>type them</b> in the chat and I will answer them.
            """

    if bot_dict[user_id].use_active_dialogue_manager:
        followup = bot_dict[user_id].get_suggested_method()
    else:
        followup = []
    # Create message dict to return ({isUser: false, feedback: false, text: initial_prompt, id: 1000})
    result_dict["initial_message"] = {
        "isUser": False,
        "feedback": False,
        "text": prompt,
        "id": 1000,
        "followup": followup
        # [{"id": "shapAllFeatures", "question": "Would you like to see the feature contributions?"}]
    }

    if user_study_group == "static":
        # Get the explanation report
        static_report = bot_dict[user_id].get_explanation_report()
        static_report["instance_type"] = bot_dict[user_id].instance_type_naming
        result_dict["static_report"] = static_report
    return result_dict


@bp.route('/get_test_datapoint', methods=['GET'])
def get_test_datapoint():
    """
    Get a new datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    return get_datapoint(user_id, "test")


@bp.route('/get_final_test_datapoint', methods=['GET'])
def get_final_test_datapoint():
    """
    Get a final test datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    return get_datapoint(user_id, "final_test")


@bp.route('/get_intro_test_datapoint', methods=['GET'])
def get_intro_test_datapoint():
    """
    Get a final test datapoint from the dataset.
    """
    user_id = request.args.get("user_id")
    return get_datapoint(user_id, "intro_test")


@bp.route("/get_user_correctness", methods=['GET'])
def get_user_correctness():
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    bot = bot_dict[user_id]
    correctness_string = bot.get_user_correctness()
    response = {"correctness_string": correctness_string}
    return response


@bp.route("/log_feedback", methods=['POST'])
def log_feedback():
    """Logs feedback"""
    feedback = request.data.decode("utf-8")
    app.logger.info(feedback)
    split_feedback = feedback.split(" || ")

    message = f"Feedback formatted improperly. Got: {split_feedback}"
    assert split_feedback[0].startswith("MessageID: "), message
    assert split_feedback[1].startswith("Feedback: "), message
    assert split_feedback[2].startswith("Username: "), message

    message_id = split_feedback[0][len("MessageID: "):]
    feedback_text = split_feedback[1][len("Feedback: "):]
    username = split_feedback[2][len("Username: "):]

    logging_info = {
        "id": message_id,
        "feedback_text": feedback_text,
        "username": username
    }

    TTM_Bot.log(logging_info)
    return ""


@bp.route('/finish', methods=['DELETE'])
def finish():
    """
    Finish the experiment.
    """
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    # Remove the bot from the dict
    try:
        bot_dict.pop(user_id)
    except KeyError:
        print(f"User {user_id} sent finish again, but the Bot was not in the dict.")
        return "200 OK"
    print(f"User {user_id} finished the experiment. And the Bot was removed from the dict.")
    return "200 OK"


@bp.route("/sample_prompt", methods=["Post"])
def sample_prompt():
    """Samples a prompt"""
    data = json.loads(request.data)
    action = data["action"]
    username = data["thisUserName"]

    prompt = sample_prompt_for_action(action,
                                      TTM_Bot.prompts.filename_to_prompt_id,
                                      TTM_Bot.prompts.final_prompt_set,
                                      real_ids=TTM_Bot.conversation.get_training_data_ids())

    logging_info = {
        "username": username,
        "requested_action_generation": action,
        "generated_prompt": prompt
    }
    TTM_Bot.log(logging_info)

    return prompt


@bp.route("/set_user_prediction", methods=['POST'])
def set_user_prediction():
    """Set the user prediction."""
    user_id = request.args.get("user_id")
    data = json.loads(request.data)
    user_prediction = data["user_prediction"]
    if user_id is None:
        user_id = "TEST"
    bot = bot_dict[user_id]
    bot.set_user_prediction(user_prediction)
    return "200 OK"


@bp.route("/get_proceeding_okay", methods=['GET'])
def get_proceeding_okay():
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    bot = bot_dict[user_id]
    proceeding_okay, follow_up_questions, response_text = bot.get_proceeding_okay()
    # Make it a message dict
    message = {
        "isUser": False,
        "feedback": True,
        "text": response_text,
        "id": 1000,
        "followup": follow_up_questions,
        "reasoning": "",
    }
    return {"proceeding_okay": proceeding_okay, "message": message}


@bp.route("/get_response_clicked", methods=['POST'])
def get_bot_response_clicked():
    """Load the box response."""
    user_id = request.args.get("user_id")
    if user_id is None:
        user_id = "TEST"
    if request.method == "POST":
        app.logger.info("generating the bot response")
        try:
            data = json.loads(request.data)
            question_id = data["question"]
            feature_id = data["feature"]
            response = bot_dict[user_id].update_state_experiment(question_id=question_id, feature_id=feature_id)
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
            feature_id = None
            question_id = None

        if bot_dict[user_id].use_active_dialogue_manager:
            followup = bot_dict[user_id].get_suggested_method()
        else:
            followup = []

        message_dict = {
            "isUser": False,
            "feedback": True,
            "text": response[0],
            "id": question_id,
            "feature_id": feature_id,
            "followup": followup,
            "reasoning": response[3]
        }

        return jsonify(message_dict)


@bp.route("/get_response_nl", methods=['POST'])
async def get_bot_response_from_nl():
    """Load the box response."""
    user_id = request.args.get("user_id")
    if request.method == "POST":
        app.logger.info("generating the bot response")
        try:
            data = json.loads(request.data)
            response, question_id, feature_id, reasoning = await bot_dict[user_id].update_state_from_nl(
                user_input=data["message"])
            if bot_dict[user_id].use_active_dialogue_manager:
                followup = bot_dict[user_id].get_suggested_method()
            else:
                followup = []
        except Exception as ext:
            app.logger.info(f"Traceback getting bot response: {traceback.format_exc()}")
            app.logger.info(f"Exception getting bot response: {ext}")
            response = "Sorry! I couldn't understand that. Could you please try to rephrase?"
            question_id = None
            feature_id = None
            followup = []
            reasoning = ""

        message_dict = {
            "isUser": False,
            "feedback": True,
            "text": response,
            "id": question_id,
            "feature_id": feature_id,
            "followup": followup,
            "reasoning": reasoning
        }
        return jsonify(message_dict)


app = Flask(__name__)
app.register_blueprint(bp, url_prefix=args.baseurl)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == "__main__":
    # clean up storage file on restart
    app.logger.info(f"Launching app from config: {args.config}")
    app.run(debug=False, port=4455, host='0.0.0.0', use_reloader=False)
