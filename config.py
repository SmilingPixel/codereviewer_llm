import json
import os


class GlobalConfig:
    """
    Global configuration class for the application.
    """
    do_train = True
    do_test = False
    device = 'cuda'

    @classmethod
    def load_from_file(cls, filepath):
        if not filepath or not os.path.isfile(filepath):
            return
        with open(filepath, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cls, k):
                setattr(cls, k, v)


class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    model_id = ''
    dataset_path = ''
    output_dir = 'output'
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 1

    @classmethod
    def load_from_file(cls, filepath):
        if not filepath or not os.path.isfile(filepath):
            return
        with open(filepath, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cls, k):
                setattr(cls, k, v)


class TestConfig:
    """
    Configuration class for testing parameters.
    """
    model_id = ''
    dataset_path = ''
    output_dir = 'output'

    @classmethod
    def load_from_file(cls, filepath):
        if not filepath or not os.path.isfile(filepath):
            return
        with open(filepath, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
