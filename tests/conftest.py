import pytest

from app.model.model import config
from app.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input__test_data():
    return load_dataset(file_name=config.app_config.test_data_file)

@pytest.fixture()
def sample_input__train_data():
    return load_dataset(file_name=config.app_config.training_data_file)    