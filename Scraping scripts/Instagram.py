# -*- coding: utf-8 -*-
"""Instagram.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rzDtIoNiZZIKCurQaEE-jAoKlYIQpUIB
"""

!pip install instaloader

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/My Drive/Colab Notebooks/

# Import the module
import instaloader

# Create an instance of Instaloader class
loader = instaloader.Instaloader(compress_json=False)

# Enter your Instagram handle and password
ACCOUNT = ''
PASSWORD = ''

# Upon successful authentication, you should see a message saying Authentication OK.
# Otherwise, check your login details
try:
    loader.login(ACCOUNT, PASSWORD)
    print("Authentication OK")
except:
    print("Error during authentication")

import datetime

HASHTAG = "shn"

# Create a Hashtag instance from a given hashtag name
hashtag = instaloader.Hashtag.from_name(loader.context, HASHTAG)

# Load posts with defined hashtag into a generator object
loaded_posts = hashtag.get_all_posts()

# To download each posts, we have to iterate over the generator object 
for cnt_post, post in enumerate(loaded_posts):
    # if cnt_post % 50 == 0:
    #   time.sleep(10)
    if datetime.datetime(2020, 3, 1) <= post.date_local <= datetime.datetime(2020, 8, 24):
      try:  
          loader.download_post(post, target="#"+hashtag.name)

      except:
          print("\nError in downloading. Process halted.") 
          break

# 'mypath' variable can be changed to your local path or Google Drive path
mypath = "."

# folder name where JSON metadata files are stored e.g, who
folder_name='#shn'

# set the path of JSON files
json_path = f'{mypath}/{folder_name}/'
json_path

import os, json
import pprint

# retrieve all filenames with .json extension located in the folder that you have provided
json_files = [filename for filename in os.listdir(json_path) if filename.endswith('.json')]

# iterate through the list of JSON files
for js in json_files:
        
    # open and read each json file
    with open(os.path.join(json_path, js)) as json_file:
        json_text = json.load(json_file)
        #pprint.pprint(json.text)
        
        try:
            # extract Instagram post information
            unix_timestamp = json_text['node']['taken_at_timestamp']
            timestamp = datetime.datetime.fromtimestamp(unix_timestamp)

            username = json_text['node']['owner']['username']
            image_desc, text = '', ''
            
            try:
                image_desc = json_text['node']['accessibility_caption']
            except:
                pass
            try:
                text = json_text['node']['edge_media_to_caption']['edges'][0]['node']['text']
            except:
                pass
            
            print (f'Timestamp: {timestamp}')
            print (f'Username: @{username}')
            print (f'Image description: {image_desc}')
            print (f'Caption: {text}')
            print('='*80)
        except:
            continue

import os, json
import pandas as pd

def convert_json_to_df(path_to_json):

    # retrieve all filenames with .json extension located in path_to_json
    json_files = [filename for filename in os.listdir(path_to_json) if filename.endswith('.json')]
    
    ## initialise list to store post details
    post_list = []

    # iterate through each json file
    for js in json_files:
        
        # posts with comments have json files for the comments, we will pass it.        
        if "comments" in js:
            continue
        
        # open and read each json file
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)

            # extract Instagram post information
            unix_timestamp = json_text['node']['taken_at_timestamp']
            timestamp = datetime.datetime.fromtimestamp(unix_timestamp)

            username = json_text['node']['owner']['username']
            image_desc, text, location = '', '',''

            postid = json_text['node']['id']
            fullname = json_text['node']['owner']['full_name']
            likes = int(json_text['node']['edge_liked_by']['count'])
            comments = int(json_text['node']['edge_media_to_comment']['count'])
            try:
                location = json_text['node']['location']['name']
            except:
                pass
        
            try:
                image_desc = json_text['node']['accessibility_caption']
            except:
                pass
            try:
                text = json_text['node']['edge_media_to_caption']['edges'][0]['node']['text']
            except:
                pass

            # append post_list with extracted post information
            post_list.append([fullname, postid, timestamp, username, image_desc, text, location, likes, comments])
            
    # populate dataframe with list of tweets
    df = pd.DataFrame(data=post_list,columns=['fullname', 'postid', 'timestamp','username','image_desc','text','location','likes','comments'])
    
    return df

pd.set_option('display.max_colwidth', 150)

# Call the function to combine and convert JSON files found in the json_path into a dataframe
df = convert_json_to_df(json_path)
df.head()

from google.colab import files
df.to_csv('shn.csv')
files.download('shn.csv')



