# Import packages
# pip install praw # uncomment and install if need be
import praw
import pandas as pd

# Access reddit API PRAW
reddit = praw.Reddit(client_id='PUMTWg7cm-mhKQ',
                         client_secret='f_SDu_F3C5Z4epNT-RKMMfY9KqlEOQ',
                         user_agent='smt203',
                         username='smt203css',
                         password='ilovesmt203!haha')

# Choose subreddit you want to scrape the data from
subreddit = reddit.subreddit("Singapore") 

# Manual list of URLS (reddit posts)
list_of_urls = [
    "https://www.reddit.com/r/singapore/comments/gn4e53/rsingapore_keeping_the_neverending_naturally/",
    "https://www.reddit.com/r/singapore/comments/gux3wc/rsingapore_postcircuit_breaker_phase_i_edition/",
    "https://www.reddit.com/r/singapore/comments/hbi1lk/rsingapore_postcircuit_breaker_phase_ii_edition/",
    "https://www.reddit.com/r/singapore/comments/fv83dd/rsingapore_april_covid19_and_circuit_breaker/",
    "https://www.reddit.com/r/singapore/comments/g0c1dx/rsingapore_april_covid19_and_circuit_breaker/",
    "https://www.reddit.com/r/singapore/comments/ga9n0l/rsingapore_april_covid19_and_circuit_breaker/",
    "https://www.reddit.com/r/singapore/comments/g5scys/rsingapore_april_covid19_and_circuit_breaker/"
]

# Initialise dictionaries (you can check the documentation if you want to add more variables, but I think these are the important ones we will need)
# Note: comment hierarchy is preserved, should there be a need to access that information...
post_dict = {
        "title" : [],
        "name" : [],
        "score" : [],
        "id" : [],
        "url" : [],
        "num_comments": [],
        "created_utc" : [],
        "upvote_ratio" : []
}
comments_dict = {
    "comment_id" : [],
    "comment_parent_id" : [],
    "comment_body" : [],
    "comment_link_id" : [],
    "created_utc" : [],
    "author" : [],
    "score" : []
}

# Iterate through list of URLS
for url in list_of_urls:
    # Initialise PRAW submission (individual post) model based on given url
    submission = reddit.submission(url=url)

    # Append post information to dictionary
    post_dict["title"].append(submission.title)
    post_dict["name"].append(submission.name)
    post_dict["score"].append(submission.score)
    post_dict["id"].append(submission.id)
    post_dict["url"].append(submission.url)
    post_dict["num_comments"].append(submission.num_comments)
    post_dict["upvote_ratio"].append(submission.upvote_ratio)
    post_dict["created_utc"].append(submission.created_utc)
    
    
    # Append comment information to comments dictionary
    # Note: replace_more method removes at most 32 instances (more than enough, I think) of MoreComments objects from the CommentForest object
    # I.e. it allows us to see all the comments at once
    submission.comments.replace_more(limit = None)
    for comment in submission.comments.list():
        comments_dict["author"].append(comment.author)
        comments_dict["score"].append(comment.score)
        comments_dict["comment_id"].append(comment.id)
        comments_dict["comment_parent_id"].append(comment.parent_id)
        comments_dict["comment_link_id"].append(comment.link_id)
        comments_dict["comment_body"].append(comment.body)
        comments_dict["created_utc"].append(comment.created_utc)
    
# Once iteration complete, create dataframe from dictionaries and save to CSVs (will save in same directory as this .py file)
# Note: Tables can be merged on comment_link_id and id to retrive post data on each comment
post_comments = pd.DataFrame(comments_dict)
post_comments.to_csv("Comments.csv")

post_data = pd.DataFrame(post_dict)
post_data.to_csv("Posts.csv")