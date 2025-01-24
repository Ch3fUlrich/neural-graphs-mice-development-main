"""Configuration module for the notebooks

Works mainly to setup data file paths

"""

import os

# Path to the user's home directory
# HOME = os.environ['HOME']

# Path to the dataset directory containing the DON-*
# folders. By default, it is pointing to a variable defined in the user's
# bash_profile (or zshrc) file. If the dataset is in your local computer,
# under your user's home folder, you could use the HOME
# variable to help indicate the data folder. E.g.:
# DATA_DIR = os.path.join(HOME, 'path', 'to', 'dataset')
try:
    DATA_DIR = os.environ["NGMD_DATA"]
except KeyError:
    print(
        "There is no environment variable named NGMD_DATA. "
        "Try setting DATA_DIR directly or assign first a "
        "value to variable NGMD_DATA in your bash_profile (or zshrc) file."
    )
    DATA_DIR = r"Y:\Users\Sergej\Steffen_Experiments\rodrigo"

# Path to the pre-trained models directory. By default, it is pointing to a
# variable defined in the user's bash_profile (or zshrc) file. If the dataset
# is in your local computer, under your user's home folder, you could use the
# HOME variable to help indicate the data folder. E.g.:
# DATA_DIR = os.path.join(HOME, 'path', 'to', 'dataset')
try:
    MODELS_DIR = os.environ["NGMD_MODELS"]
except KeyError:
    print(
        "There is no environment variable named NGMD_MODELS. "
        "Try setting MODELS_DIR directly or assign first a "
        "value to variable NGMD_MODELS in your bash_profile (or zshrc) file."
    )
    MODELS_DIR = ""
