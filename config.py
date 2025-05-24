class GlobalConfig:
    """
    Global configuration class for the application.
    """
    do_train = True
    do_test = False


class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    model_id = ''
    dataset_path = ''
    output_dir = 'output'


class TestConfig:
    """
    Configuration class for testing parameters.
    """
    model_id = ''
    dataset_path = ''
    output_dir = 'output'


        