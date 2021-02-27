#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install instaloader')


# In[1]:


import datetime
import os, json
import pprint
import os, json
import pandas as pd


# In[10]:


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


# **USING PROFILES**

# In[13]:


def get_profile_posts(username):
    # Load the hashtag object into a variable
    profile = instaloader.Profile.from_username(loader.context, username)
    
    # Get top posts in a generator
    posts = profile.get_posts()
        
    for post in posts:
        if datetime.datetime(2020, 1, 1) <= post.date_local <= datetime.datetime(2020, 4 , 7):
            if 'COVID-19' in post.caption:
                try:
                    loader.download_post(post, target=f"@{profile.username}")

                except:
                    print("Error in downloading post")
                    break # If there are any errors, we break out of the loop


# In[14]:


USERNAME = 'leehsienloong'

get_profile_posts(USERNAME)


# **USING HASHTAGS**

# In[ ]:


HASHTAG = "shn"

# Create a Hashtag instance from a given hashtag name
hashtag = instaloader.Hashtag.from_name(loader.context, HASHTAG)

# Load posts with defined hashtag into a generator object
loaded_posts = hashtag.get_all_posts()

# To download each posts, we have to iterate over the generator object 
for post in enumerate(loaded_posts):
    if datetime.datetime(2020, 3, 1) <= post.date_local <= datetime.datetime(2020, 8, 24):
        try:  
            loader.download_post(post, target="#"+hashtag.name)
            time.sleep(3)

        except:
            print("\nError in downloading. Process halted.") 
            break


# **RETRIEVE ALL FILENAMES WITH .JSON EXTENSION LOCATED IN THE FOLDER**

# In[2]:


# 'mypath' variable can be changed to your local path or Google Drive path
mypath = "."

# folder name where JSON metadata files are stored e.g, who
folder_name='@leehsienloong'

# set the path of JSON files
json_path = f'{mypath}/{folder_name}/'
json_path


# In[3]:


json_files = [filename for filename in os.listdir(json_path) if filename.endswith('.json')]

# iterate through the list of JSON files
for js in json_files:
    
    with open(os.path.join(json_path, js)) as json_file:
        json_text = json.load(json_file)
        # print(json_text)
        # pprint.pprint(json_text)    
    
        # Extract comments into list
        comments = []
        for i in json_text: 
            if 'text' in i: 
                comments += [i['text']]
        print (f'Comments: {comments}')

    with open(os.path.join(json_path, js)) as json_file:
        json_text = json.load(json_file)
        # print(json_text)
        # pprint.pprint(json_text)     

        try:
            # extract Instagram post information
            unix_timestamp = json_text['node']['taken_at_timestamp']
            timestamp = datetime.datetime.fromtimestamp(unix_timestamp)
            username = json_text['node']['owner']['username']
            postid, fullname, image_desc, text  = '', '', '', ''

            try:
                postid = json_text['node']['id']
            except:
                pass

            try:
                fullname = json_text['node']['owner']['full_name']
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

            print (f'Timestamp: {timestamp}')
            print (f'Username: @{username}')
            print (f'Postid: {postid}')
            print (f'Fullname: {fullname}')
            print (f'Image description: {image_desc}')
            print (f'Caption: {text}')
            print('='*80)

        except:
            continue


# **CONVERT JSON FILES TO DATAFRAME**

# In[4]:


def convert_json_to_df(path_to_json):

    # retrieve all filenames with .json extension located in path_to_json
    json_files = [filename for filename in os.listdir(path_to_json) if filename.endswith('.json')]
    
    ## initialise list to store post details
    post_list = []

    # iterate through each json file
    for js in json_files:            
        
        comments = []
        if "comments" not in js:
            
            # open and read each json file
            with open(os.path.join(path_to_json, js)) as json_file:
                json_text = json.load(json_file)

                # extract Instagram post information
                unix_timestamp = json_text['node']['taken_at_timestamp']
                timestamp = datetime.datetime.fromtimestamp(unix_timestamp)

                username = json_text['node']['owner']['username']
                postid, fullname, image_desc, text = '', '', '', ''

                postid = json_text['node']['id']
                fullname = json_text['node']['owner']['full_name']

                try:
                    image_desc = json_text['node']['accessibility_caption']
                except:
                    pass
                try:
                    text = json_text['node']['edge_media_to_caption']['edges'][0]['node']['text']
                except:
                    pass

        else:
            
            # open and read each json file
            with open(os.path.join(path_to_json, js)) as json_file:
                json_text = json.load(json_file)

                # Extract comments into list
                for i in json_text: 
                    if 'text' in i: 
                        comments += [i['text']]
            
            # append post_list with extracted post information
            post_list.append([fullname, postid, timestamp, username, image_desc, text, comments])
            
    # populate dataframe with list of tweets
    df = pd.DataFrame(data=post_list,columns=['fullname', 'postid', 'timestamp','username','image_desc','text', 'comments'])
    
    return df


# In[5]:


pd.set_option('display.max_colwidth', 150)


# In[6]:


# Call the function to combine and convert JSON files found in the json_path into a dataframe
df = convert_json_to_df(json_path)
df.head()


# **EXPORT DATAFRAME TO CSV**

# In[8]:


df.to_csv('cna_covid19.csv')


# In[ ]:




