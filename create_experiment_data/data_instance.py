import pandas as pd


class DataInstance:
    def __init__(self, id, values, model_prediction, true_label):
        self.id = id
        self.values = values
        self.model_prediction = model_prediction
        self.true_label = true_label

    def _prepare_instance_data(self, instance, model_prediction, true_label):
        return DataInstance(instance['id'], instance['values'], model_prediction, true_label)
