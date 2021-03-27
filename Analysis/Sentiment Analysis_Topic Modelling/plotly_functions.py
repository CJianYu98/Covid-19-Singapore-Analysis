import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
from plotly.offline import plot, iplot, init_notebook_mode
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parent_path = "/Users/joshuawong/Documents/GitHub/Covid-19-Singapore-Analysis"
# parent_path = "C:/Users/user/Documents/GitHub/Covid-19-Singapore-Analysis"

def get_data(policy, timeframe):
    df = pd.read_csv(f"{parent_path}/Data/Sentiment Data/{policy}_valuable_opinions.csv")

    # Standardising datetime format
    dates = []
    for row in df['Comment Datetime']:
        if '/' in row:
            if '/' in row[-4:]:
                date = dt.datetime.strptime(row, "%d/%m/%y")
            else: 
                date = dt.datetime.strptime(row, "%d/%m/%Y")
            dates.append(date.date())
        elif '-' in row:
            date = dt.datetime.strptime(row, "%Y-%m-%d")
            dates.append(date.date())

    df['Comment Datetime'] = dates

    # Truncate according to the policy timeframe
    start, end = pd.to_datetime([timeframe[0], timeframe[1]], format='%d%b%Y')
    df_within_date = df[(df['Comment Datetime'] >= start) & (df['Comment Datetime'] <= end)]
    df_within_date = df_within_date[['Comment Datetime', 'Vader_compound_score']]
    return df

def sentiment_with_comments_static(policy, timeframe):
    df_within_date = get_data(policy, timeframe)

    # Finding the mean vader sentiment score per day
    grouped_date = df_within_date.groupby(by=["Comment Datetime"]).mean()
    grouped_date.reset_index(inplace=True)

    # Finding the moving average (7-day)
    grouped_date['SMA_7'] = grouped_date.Vader_compound_score.rolling(7, min_periods=1).mean()

    # Finding the number of comments per day
    grouped_date_counts = df_within_date.groupby(by=["Comment Datetime"]).count()
    grouped_date_counts.reset_index(inplace=True)
    grouped_date['Comment Count'] = grouped_date_counts['Vader_compound_score']

    # Plotting the figure 
    fig = make_subplots(specs=[[{"secondary_y": True}]]) # NEW
    fig.add_trace(go.Scatter(x = grouped_date['Comment Datetime'], y = grouped_date['SMA_7'], mode='lines', 
                  name="Vader 7-day MA"), secondary_y=False)
    fig.add_trace(go.Scatter(x = grouped_date['Comment Datetime'], y = grouped_date['Vader_compound_score'], mode='lines', 
                  name="Vader Compound Score", opacity=.5), secondary_y=False)
    fig.add_bar(x = grouped_date['Comment Datetime'], y = grouped_date['Comment Count'],
                name="Comment Count", secondary_y=True)

    #fig.update_layout(width = 1200, height=800)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      title_text=f"Timeseries Sentiment Analysis of {policy} with Daily Number of Comments",
                      yaxis_range=[-2,1])

    max_num_comments = grouped_date['Comment Count'].max()
    fig.update_layout(yaxis2_range=[0,max_num_comments*2.5])
    fig.update_layout(width = 1200, height=800)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Vader Compound Score')
    fig.update_yaxes(title_text='Comment Count', secondary_y=True)

    fig.write_image(f"timeseries sentiment plots/{policy}_with_comments.png", engine="kaleido")
    # fig.show()
    print(policy, "(with comments, static) is done!")

def sentiment_with_comments_html(policy, timeframe):
    df_within_date = get_data(policy, timeframe)

    # Finding the mean vader sentiment score per day
    grouped_date = df_within_date.groupby(by=["Comment Datetime"]).mean()
    grouped_date.reset_index(inplace=True)

    # Finding the moving average (7-day)
    grouped_date['SMA_7'] = grouped_date.Vader_compound_score.rolling(7, min_periods=1).mean()

    # Finding the number of comments per day
    grouped_date_counts = df_within_date.groupby(by=["Comment Datetime"]).count()
    grouped_date_counts.reset_index(inplace=True)
    grouped_date['Comment Count'] = grouped_date_counts['Vader_compound_score']

    # Plotting the figure 
    fig = make_subplots(specs=[[{"secondary_y": True}]]) # NEW
    fig.add_trace(go.Scatter(x = grouped_date['Comment Datetime'], y = grouped_date['SMA_7'], mode='lines', 
                  name="Vader 7-day MA"), secondary_y=False)
    fig.add_trace(go.Scatter(x = grouped_date['Comment Datetime'], y = grouped_date['Vader_compound_score'], mode='lines', 
                  name="Vader Compound Score", visible='legendonly', opacity=.5), secondary_y=False)

    fig.add_bar(x = grouped_date['Comment Datetime'], y = grouped_date['Comment Count'],
                name="Comment Count", secondary_y=True, visible='legendonly')

    #fig.update_layout(width = 1200, height=800)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      title_text=f"Timeseries Sentiment Analysis of {policy} with Daily Number of Comments",
                      yaxis_range=[-1,1])

    max_num_comments = grouped_date['Comment Count'].max()
    fig.update_layout(yaxis2_range=[0,max_num_comments*2.5])
    fig.update_layout(width = 1200, height=800)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Vader Compound Score')
    fig.update_yaxes(title_text='Comment Count', secondary_y=True)

    fig.write_html(f"timeseries sentiment plots/{policy}_with_comments.html")
    # fig.show()
    print(policy, "(with comments, html) is done!")

def sentiment_with_cases_static(policy, timeframe):
    df_within_date = get_data(policy, timeframe)

    # Getting daily confirmed cases
    daily_cases = pd.read_excel(f"{parent_path}/Analysis/Sentiment Analysis_Topic Modelling/Covid-19 SG.xlsx") 
    daily_cases.reset_index(inplace=True)
    daily_confirmed = daily_cases.iloc[:, 1:3]

    # Finding the mean vader sentiment score per day
    grouped_date = df_within_date.groupby(by=["Comment Datetime"]).mean()
    grouped_date.reset_index(inplace=True)

    # Finding the moving average (7-day)
    grouped_date['SMA_7'] = grouped_date.Vader_compound_score.rolling(7, min_periods=1).mean()

    # Finding the number of comments per day
    grouped_date_counts = df_within_date.groupby(by=["Comment Datetime"]).count()
    grouped_date_counts.reset_index(inplace=True)
    grouped_date['Comment Count'] = grouped_date_counts['Vader_compound_score']

    # Setting the number of cases within the policy timeframe 
    daily_confirmed_within_timeframe = daily_confirmed.loc[(daily_confirmed["Date"]>= timeframe[0]) & (daily_confirmed["Date"]<= timeframe[1])]

    # Plotting the figure 
    fig = make_subplots(specs=[[{"secondary_y": True}]]) # NEW
    fig.add_trace(go.Scatter(x = grouped_date['Comment Datetime'], y = grouped_date['SMA_7'], mode='lines', 
                  name="Vader 7-day MA"), secondary_y=False)
    fig.add_trace(go.Scatter(x = grouped_date['Comment Datetime'], y = grouped_date['Vader_compound_score'], mode='lines', 
                  name="Vader Compound Score", opacity=.5), secondary_y=False)

    fig.add_bar(x = daily_confirmed_within_timeframe["Date"], y = daily_confirmed_within_timeframe["Daily Confirmed "],
                name="Cases Count", secondary_y=True)

    #fig.update_layout(width = 1200, height=800)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      title_text=f"Timeseries Sentiment Analysis of {policy} with Daily Number of Cases",
                      yaxis_range=[-2,1], 
                      yaxis2_range=[0,1426*2])
                      
    fig.update_layout(width = 1200, height=800)
    fig.update_xaxes(title_text='Date', ticks="outside", showgrid=False)
    fig.update_yaxes(title_text='Vader Compound Score', ticks="outside", showgrid=False)
    fig.update_yaxes(title_text='Number of Cases', ticks="outside", showgrid=False, secondary_y=True)

    fig.write_image(f"timeseries sentiment plots/{policy}_with_cases.png", engine="kaleido")
    # fig.show()
    print(policy, "(with cases, static) is done!")

def sentiment_with_cases_html(policy, timeframe):
    df_within_date = get_data(policy, timeframe)

    # Getting daily confirmed cases
    daily_cases = pd.read_excel(f"{parent_path}/Analysis/Sentiment Analysis_Topic Modelling/Covid-19 SG.xlsx") 
    daily_cases.reset_index(inplace=True)
    daily_confirmed = daily_cases.iloc[:, 1:3]

    # Finding the mean vader sentiment score per day
    grouped_date = df_within_date.groupby(by=["Comment Datetime"]).mean()
    grouped_date.reset_index(inplace=True)

    # Finding the moving average (7-day)
    grouped_date['SMA_7'] = grouped_date.Vader_compound_score.rolling(7, min_periods=1).mean()

    # Finding the number of comments per day
    grouped_date_counts = df_within_date.groupby(by=["Comment Datetime"]).count()
    grouped_date_counts.reset_index(inplace=True)

    grouped_date['Comment Count'] = grouped_date_counts['Vader_compound_score']

    # Setting the number of cases within the policy timeframe 
    daily_confirmed_within_timeframe = daily_confirmed.loc[(daily_confirmed["Date"]>= timeframe[0]) & (daily_confirmed["Date"]<= timeframe[1])]

    # Plotting the figure 
    fig = make_subplots(specs=[[{"secondary_y": True}]]) # NEW
    fig.add_trace(go.Scatter(x = grouped_date['Comment Datetime'], y = grouped_date['SMA_7'], mode='lines', 
                  name="Vader 7-day MA"), secondary_y=False)
    fig.add_trace(go.Scatter(x = grouped_date['Comment Datetime'], y = grouped_date['Vader_compound_score'], mode='lines', 
                  name="Vader Compound Score", opacity=.5,visible='legendonly'), secondary_y=False)
    fig.add_bar(x = daily_confirmed_within_timeframe["Date"], y = daily_confirmed_within_timeframe["Daily Confirmed "],
                name="Cases Count", secondary_y=True, visible='legendonly')

    #fig.update_layout(width = 1200, height=800)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      title_text=f"Timeseries Sentiment Analysis of {policy} with Daily Number of Comments",
                      yaxis_range=[-2,1],
                      yaxis2_range=[0,1426*2])
                      
    fig.update_layout(width = 1200, height=800)
    fig.update_xaxes(title_text='Date', ticks="outside", showgrid=False)
    fig.update_yaxes(title_text='Vader Compound Score', ticks="outside", showgrid=False)
    fig.update_yaxes(title_text='Number of Cases', ticks="outside", showgrid=False, secondary_y=True)

    fig.write_html(f"timeseries sentiment plots/{policy}_with_cases.html")
    # fig.show()
    print(policy, "(with cases, html) is done!")

def emotion_count(policy):
    anger_df = pd.read_csv(f"{parent_path}/Data/Sentiment Data/{policy}_valuable_anger.csv")
    fear_df = pd.read_csv(f"{parent_path}/Data/Sentiment Data/{policy}_valuable_fear.csv")
    joy_df = pd.read_csv(f"{parent_path}/Data/Sentiment Data/{policy}_valuable_joy.csv")
    neutral_df = pd.read_csv(f"{parent_path}/Data/Sentiment Data/{policy}_valuable_neu.csv")
    sad_df = pd.read_csv(f"{parent_path}/Data/Sentiment Data/{policy}_valuable_sad.csv")

    emotions = ['anger', 'fear','sad','neutral','joy']
    counts = [len(anger_df), len(fear_df), len(sad_df), len(neutral_df), len(joy_df)] 

    fig = go.Figure([go.Bar(x=emotions, y=counts, text=counts)])
    fig.update_layout(title_text=f"Emotion Counts for {policy}")
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    # fig.show()
    fig.write_image(f"timeseries sentiment plots/Emotions/{policy}_emotions.png", engine="kaleido")

    print(policy, "(emotion count) is done!")
