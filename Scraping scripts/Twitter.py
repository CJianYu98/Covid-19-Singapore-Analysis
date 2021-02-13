# Import packages
import twint
import time
import json

# Set up configurations
c = twint.Config()    
c.Store_csv = True # Store to json
c.Since = "2020-01-01 00:00:00" # Set start date for collection
c.Until = "2021-02-01 00:00:00" # Set end date for collection
c.Retweets = True # Include retweets done by user
c.Lang = "en" # Set language
c.Limit = 100000 # Set tweet limit to 10k 
c.Near = "Singapore" # Set geograpic location to near Singapore
c.Search = "panic buying" # Set search term
c.Output = "./panic buying 010120 - 010221.csv" # Save output in current directory containing python script

# Run
twint.run.Search(c)

