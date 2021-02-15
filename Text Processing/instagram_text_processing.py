import os
from text_processing_functions import *

keywords = [] #
stopwords_list = stopwords_ls(keywords)


mypath = ".."
folder_name='Data/Instagram Data/Raw Data'
file_path = f'{mypath}/{folder_name}/'
insta_page_folder = [folder for folder in os.listdir(file_path) if (folder != 'Extras' and folder != '.DS_Store')]


"""
    Code for merging instagram excel files into one based on the same keyword search at the same Instagram Acc. Un-comment to run if needed. 
"""
for insta_page in insta_page_folder:
    keyword_folder = [folder for folder in os.listdir(f'{file_path}{insta_page}') if (folder != '.DS_Store' and folder != 'Article list.xlsx')]

    folder_path = f'../Data/Instagram Data/Cleaned Data/{insta_page}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    for keyword in keyword_folder:
        frames = []
        xlsx_files = [filename for filename in os.listdir(f'{file_path}{insta_page}/{keyword}')]

        for xlsx_file in xlsx_files:
            # print(f'{file_path}{insta_page}/{keyword}/{xlsx_file}')
            xlsx_df = pd.read_excel(f'{file_path}{insta_page}/{keyword}/{xlsx_file}', engine='openpyxl')
            frames.append(xlsx_df)
        
        df = pd.concat(frames)

        output = instagram_text_processing(df, stopwords_list)

        output.to_csv(f'../Data/Instagram Data/Cleaned Data/{insta_page}/{keyword}.csv')
