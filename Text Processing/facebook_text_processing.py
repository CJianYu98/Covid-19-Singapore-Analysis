import os
import pandas as pd
from text_processing_functions import *

mypath = ".."
folder_name='Data/Twitter Data/Raw Data'
file_path = f'{mypath}/{folder_name}/'
folders = [folder for folder in os.listdir(file_path) if folder != '.DS_Store']