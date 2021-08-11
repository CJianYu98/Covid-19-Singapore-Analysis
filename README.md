# Public Sentiment on Covid-19 Policies in Singapore<a name="top"></a>
This project on the analysis of COVID-19 policies in Singapore was one of the assessment components for the Singapore Management University (SMU) module SMT203 - Computational Social Science.

---

## Table of Contents
- [Introduction](#Introduction)
    - [Motivation](#Motivation)
    - [Project Description](#Project-Description)
- [Dataset](#Dataset)
    - [Data Collection](#Data-Collection)
    - [Data Processing](#Data-Processing)
- [Modelling](#Modelling)
    - [Valuable Comments Classifier (Supervised Learning - Classification)](#val_cmts_clf1)
    - [Valuable Comments Classifier (Bag of Words)](#val_cmts_clf2)
    - [Sentiment Analysis (Lexicon and Ruled Based)](#sentiment_analysis)
    - [Time Series Analysis](#Time-Series-Analysis)
    - [Topic Modelling](#Topic-Modelling)
- [Findings](#Findings)
- [Code Navigation](#Code-Navigation)
- [Contributors](#contributors)

---

## Introduction
This project on the analysis of COVID-19 policies in Singapore was one of the assessment components for the Singapore Management University (SMU) module SMT203 - Computational Social Science.

In this project, the team aims to use Natural Language Processing techniques, such as sentiment analysis and topic modelling, to discover insights regarding COVID-19 policies in Singapore from comments by online users on social media platforms. 

While the implementation of several COVID-19 related policies should not entirely hinge on public opinion but rather on medical knowledge, value in analysing opinions lies in follow-up policymaking where tweaks can be made to better address citizens’ concerns especially in the context of the ever-changing pandemic situation.

<br>

### Motivation
Public opinions are important to the government in policy making, especially in a parliamentary democracy such as Singapore’s. Apart from its role in increasing the chances of advocacy success and other political reasons, public opinions can enable the government to gather insights and feedback from the citizenry to fine-tune and follow up on various policy initiatives.

However, traditional feedback gathering methods are labour intensive and time consuming, limiting the overall usefulness when information is needed in a timely manner. In addition, social desirability bias can affect the results with respondents reporting more favourable responses. Also, co-creation is only possible if citizens themselves desire to be personally involved in the policy making process.

Due to the limitations in traditional methods of opinion gathering, there is value in seeking the opinions of the public through the comments they post on social media platforms as an alternative. Social media data provides a large, fast and cheap stream of information which enables policy makers to consider the public's opinions during policy making and monitor different perceptions of policy beneficiaries during policy implementation and evaluation. Another advantage is the flexibility of users remaining anonymous, hence mitigating the effects of social desirability bias on the comments collected.

<br>

### Project Description
Our team sought to work on COVID-19 related policies due to its contemporary and unprecedented nature. We believe that analysis of public opinion has great value in follow-up policymaking where tweaks can be made to better address citizens’ concerns. 

Specifically, the analysis aims to explore how social media data analysis can be leveraged on to enhance government decision-making in designing and implementing COVID-19 related policies in the Singapore context. Our team performed sentiment analysis, topic modelling and time series analysis.

We have focused specifically on these seven policies: 
- Circuit Breaker (CB)
- TraceTogether
- Vaccination
- Social distancing
- Foreign worker dormitory
- Economic measures
- Mandatory mask-wearing

We have focused on these five social media platforms:
- Facebook
- Instagram
- Twitter
- Reddit
- Hardwarezone

[Back To The Top](#top)

---
## Dataset
Since the dataset is too large, you may download the data from [here](https://www.kaggle.com/chenjianyu/covid19-singapore-policies-dataset).

A total of 107,096 comments were collected across the various platforms.

<br>

### Data Collection
Comments posted between 1 January 2020 to 1 March 2021 were collected. For each policy, a list of relevant keywords was generated using prior knowledge and relevant queries or topics from Google Trends. Datetime for each comment is also scrapped for time series analysis later on.

Different methods were used to collect comments for each of the social media platforms. 

Using a keyword-based approach, Python Twint package and Python Reddit API Wrapper (PRAW) were used to collect data from Twitter and Reddit respectively. 

For Instagram, Facebook and Hardwarezone, BeautifulSoup and Selenium were used, where relevant posts or threads links were manually sieved out before scraping the comments. Instagram and Facebook were scrapped based on accounts which are perceived as credible and where netizens frequently engage in discussion

<br>

### Data Pre-processing
Unnecessary columns of data are removed alongside rows with empty comment or datetime. Regex was used to remove URLs, hashtags, mentions as they contain no useful information. Comments with less than 5 words are unlikely to be insightful and are removed. Datetime was standardised across all platforms to ensure comments could be sorted by datetime to conduct time series analysis.

[Back To The Top](#top)

---
## Modelling
This section will elaborate of the different models and techniques used.

<br>

### Valuable Comment Classifier (Supervised Learning - Classification)<a name="val_cmts_clf1"></a>
Dealing with social media data means dealing with a lot of noise because comments are mostly user-generated and netizens are able to freely express their opinions. Thus, filtering of noise to extract useful comments is crucial.

Valuable comments should provide insights that are useful for analysis, to give policymakers quality information that can enhance policymaking and help them make better decisions. As such, we adopted a supervised learning approach to create a classifier, to measure the text quality.  

Features used are as follows:
- Comment length
- Comment quality (based on lexical features)
- Number of verbs
- Number of discourse relations
- Number of pronouns
- Relevance score to a given policy

Various classification models were tested to find the most optimal model.

<br>

### Valuable Comment Classifier (Bag of Words) <a name="val_cmts_clf2"></a>
Bag of words approach is also expored to utilize textual information to classify comments into valuable and invaluable comments.

<br>

### Sentiment Analysis (Lexicon and Ruled Based) <a name="sentiment_analysis"></a>
Polarity and subjectivity of the text were calculated using Python VADER and TextBlob packages.

In addition, the team also modelled a emotion classifier which aims to classify comments into various emotions, namely:
- Joy
- Anger
- Fear
- Sadness
- Neutral

<br>

### Time Series Analysis
After sentiment score were calculated, moving averages of this score were further calculated. These moving average compound score were then used to plot various charts for analysis.

- Against number of comments 
The team has plotted the changes in sentiments across time with the total number of comments collected related to the policy of interest in helping us to track specific events that have occurred to trigger mass discussion from the public.

- Against number of daily cases
The team has plotted the change in sentiments across time against the number of Covid-19 cases to analyse the change in sentiment with relation to effectiveness of the policy implemented.

<br>

### Topic Modelling
Our team experimented with the Latent Dirichlet Allocation Mallet, which is commonly used to find topics in a corpus/text. However, the team found it challenging to identify topics from the keywords generated and it requires non-stop iteration of stopwords removal. Therefore, our team decided to use CorEx topic modelling to incorporate word-level domain knowledge through anchor words to produce the topics of interest.

CorEx is a semi-supervised topic modelling that finds words that provide the highest mutual information with the topic. Compared to the LDA Mallet, CorEx is able to produce more coherent topics, allowing us to have a more in depth understanding as compared to the general timeline sentiment analysis.

[Back To The Top](#top)

---
## Findings

![How sentiments on social media are targeted towards different issues](https://i.postimg.cc/RVG3TjVN/Picture-1.jpg)

We found that sentiments of netizens’ comments were often related to how the policy affects themselves and their way of life. For example, in early May, the announcement of tightened measures were meant to further control the spread of the virus and we expect public sentiment to increase. However, the sentiment decreased instead possibly because the tightened measures were about bubble tea shops, alcohol and other F&B stores not being able to reopen. Netizens were concerned and upset about not being able to purchase their bubble tea instead, as seen as one of the rising topics reflected in Google Trends. Sentiments could also be related to policy implementation and other details about the policy or downstream effects of the policy on other entities.

Overall, we can see that instead of discussing about the effectiveness of the policies themselves and its efficacy in reducing the number of Covid-19 cases directly, many netizens comment about the immediate impact of the policies implemented on themselves, taking a very individualistic positions instead of looking at the bigger picture and the potential effectiveness of the policies themselves. Sentiment is also likely to be affected by many external factors not relating to the specific policy. It is difficult to isolate the effects of just one policy on the sentiment expressed by netizens, and thus, change in sentiment is likely not to have any correlation to the change in the number of cases as originally hypothesized.

[Back To The Top](#top)

---
## Code Navigation
1. Analysis Folder - contains scripts and notebooks and findings for our analysis 
2. Data Preparation Folder - contains scripts and notebook for cleaning and pre processing of raw data before modelling and analysis
3. Data Scraping Folder - contains scripts for scraping web data on social media platforms
4. General EDA Folder - contains scripts and notebooks for exploratory data analysis

[Back To The Top](#top)

---
## Contributors

1. Chen Jian Yu ([Linkedin](https://www.linkedin.com/in/chen-jian-yu/), [Github](https://github.com/CJianYu98))
2. Joshua Wong Yeung Nguon ([Linkedin](https://www.linkedin.com/in/joshuawong96/), [Github](https://github.com/joshuawong96))
1. Ow Ling Jia ([Linkedin](https://www.linkedin.com/in/owlingjia/), [Github](https://github.com/owlingjia))

[Back To The Top](#top)