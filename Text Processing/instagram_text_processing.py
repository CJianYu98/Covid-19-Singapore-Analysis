import os
from text_processing_functions import *


mypath = "."
folder_name='Instagram Data'
file_path = f'{mypath}/{folder_name}/'
csv_files = [filename for filename in os.listdir(file_path) if filename.endswith('.csv')]
insta_page_folder = [filename for filename in os.listdir(file_path) if (filename.endswith('.csv') == False and filename != '.DS_Store')]


"""
    Code for merging instagram excel files into one based on the same keyword search at the same Instagram Acc. Un-comment to run if needed. 
"""
# for folder in insta_page_folder:
#     file_path_1 = file_path + folder + '/'
#     keyword_files = [filename for filename in os.listdir(file_path_1) if (filename.endswith('.xlsx') == False and filename != '.DS_Store' and filename != 'Cleaned Data')]

#     for file in keyword_files:
#         file_path_2 = file_path_1 + file + '/'

#         frames = []
#         for xlsx_file_name in os.listdir(file_path_2):
#             xlsx_file = file_path_2 + xlsx_file_name
#             xlsx_df = pd.read_excel(xlsx_file, engine='openpyxl')
#             frames.append(xlsx_df)
        
#         result = pd.concat(frames)
#         result.to_csv(f'./Instagram Data/{folder}/Cleaned Data/{file}.csv')

"""
    End of code to merge instagram excel files
"""

for folder in insta_page_folder:
    cleaned_data_path = file_path + folder + '/Cleaned Data' + '/'

    for csv_file_name in os.listdir(cleaned_data_path):
        csv_file = cleaned_data_path + csv_file_name
        csv_df = pd.read_csv(csv_file)
        filename = csv_file_name

        instagram_text_processing(csv_df, folder, filename)