import flask
import os
from flask import send_from_directory, request
import pandas as pd
import numpy as np
import re
import itertools
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify


app = flask.Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return "Hello World"



filename = "steam.csv"

df = pd.read_csv(filename,  encoding='utf-8')

#These are just quick checks to make sure the dataset looks correct
# print(df.shape)
df.head()
df.columns
#Which columns have null values?
# print(df.columns[df.isna().any()].tolist())

#How many null values per column?
df.isnull().sum()

# The function to extract years
def extract_year(date):
   year = date[:4]
   # some games do not have the info about year in the column title. So, we should take care of the case as well.
   if year.isnumeric():
      return int(year)
   else:
      return np.nan

df['year'] = df['release_date'].apply(extract_year)
df.head()

#Create score column
def create_score(row):
  pos_count = row['positive_ratings']
  neg_count = row['negative_ratings']
  total_count = pos_count + neg_count
  average = pos_count / total_count
  return round(average, 2)

def total_ratings(row):
  pos_count = row['positive_ratings']
  neg_count = row['negative_ratings']
  total_count = pos_count + neg_count
  return total_count

df['total_ratings'] = df.apply(total_ratings, axis=1)
df['score'] = df.apply(create_score, axis=1)

# Calculate mean of vote average column
C = df['score'].mean()

# Calculate the minimum number of votes required to be in the chart,
m = df['total_ratings'].quantile(0.90)

# calculate the weighted rating for each qualified game
# Function that computes the weighted rating of each game
def weighted_rating(x, m=m, C=C):
    v = x['total_ratings']
    R = x['score']
    # Calculation based on the IMDB formula
    return round((v/(v+m) * R) + (m/(m+v) * C), 2)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
df['weighted_score'] = df.apply(weighted_rating, axis=1)

#Print the top 15 games
# df[['name', 'total_ratings', 'score', 'weighted_score']].head(15)

#The reason we're adding this is for tags with multiple words, we need to connect by '-' before we split them by ' '
df['steamspy_tags'] = df['steamspy_tags'].str.replace(' ','-')
#TFIDF
df['genres'] = df['steamspy_tags'].str.replace(';',' ')
# count the number of occurences for each genre in the data set
counts = dict()
for i in df.index:
  #for each element in list (each row, split by ' ', in genres column)
  #-- we're splitting by space so tfidf can interpret the cells
   for g in df.loc[i,'genres'].split(' '):
#if element is not in counts(dictionary of genres)
      if g not in counts:
        #give genre dictonary entry the value of 1
         counts[g] = 1
      else:
        #increase genre dictionary entry by 1
         counts[g] = counts[g] + 1

#Test Genre Counts
counts.keys()
# print(counts['Action'])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz

# create an object for TfidfVectorizer
tfidf_vector = TfidfVectorizer(stop_words='english')
# apply the object to the genres column
# convert the list of documents (rows of genre tags) into a matrix
tfidf_matrix = tfidf_vector.fit_transform(df['genres'])

tfidf_matrix.shape
#The tfidf_matrix is the matrix with 27075 rows(games) and 370 columns(genres)

# print(list(enumerate(tfidf_vector.get_feature_names_out())))
# create the cosine similarity matrix
sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)
# print(sim_matrix)

# create a function to find the closest title name
def matching_score(a,b):
  #fuzz.ratio(a,b) calculates the Levenshtein Distance between a and b, and returns the score for the distance
   return fuzz.ratio(a,b)
   # exactly the same, the score becomes 100

##These functions needed to return different attributes of the recommended game titles

#Convert index to title_year
def get_title_year_from_index(index):
   return df[df.index == index]['year'].values[0]
#Convert index to title
def get_title_from_index(index):
   return df[df.index == index]['name'].values[0]
#Convert index to title
def get_index_from_title(title):
   return df[df.name == title].index.values[0]
#Convert index to score
def get_score_from_index(index):
   return df[df.index == index]['score'].values[0]
#Convert index to weighted score
def get_weighted_score_from_index(index):
   return df[df.index == index]['weighted_score'].values[0]
#Convert index to total_ratings
def get_total_ratings_from_index(index):
   return df[df.index == index]['total_ratings'].values[0]
#Convert index to platform
def get_platform_from_index(index):
  return df[df.index == index]['platforms'].values[0]

# A function to return the most similar title to the words a user type
# Without this, the recommender only works when a user enters the exact title which the data has.
def find_closest_title(title):
  #matching_score(a,b) > a is the current row, b is the title we're trying to match
   leven_scores = list(enumerate(df['name'].apply(matching_score, b=title))) #[(0, 30), (1,95), (2, 19)~~] A tuple of distances per index
   sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True) #Sorts list of tuples by distance [(1, 95), (3, 49), (0, 30)~~]
   closest_title = get_title_from_index(sorted_leven_scores[0][0])
   distance_score = sorted_leven_scores[0][1]
   return closest_title, distance_score

#find_closest_title returns only one title but I want a dropdown of the 10 closest game titles
def closest_names(title):
   leven_scores = list(enumerate(df['name'].apply(matching_score, b=title)))
   sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
   top_closest_names = [get_title_from_index(i[0]) for i in sorted_leven_scores[:10]]  #[['Team Fortress Classic', 'Deathmatch Classic', 'Counter-Strike',~~]
   return top_closest_names

# @app.route('/recommend', methods=['POST'])
def gradio_contents_based_recommender_v2(game, how_many, dropdown_option, sort_option, min_year, platform, min_score):
  #Return closest game title match
  closest_title, distance_score = find_closest_title(dropdown_option)
  #Create a Dataframe with these column headers
  recomm_df = pd.DataFrame(columns=['Game Title', 'Year', 'Score', 'Weighted Score', 'Total Ratings'])
  #Make the closest title whichever dropdown option the user has chosen
  closest_title = dropdown_option
  # print(closest_title)
  #find the corresponding index of the game title
  games_index = get_index_from_title(closest_title)
  #return a list of the most similar game indexes as a list
  games_list = list(enumerate(sim_matrix[int(games_index)]))
  #Sort list of similar games from top to bottom
  similar_games = list(filter(lambda x:x[0] != int(games_index), sorted(games_list,key=lambda x:x[1], reverse=True)))
  #Print the game title the similarity matrix is based on
  # print('Here\'s the list of games similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')
  #Only return the games that are on selected platform
  n_games = []
  for i,s in similar_games:
    if platform in get_platform_from_index(i):
      n_games.append((i,s))
  #Only return the games that are above the minimum score
  high_scores = []
  for i,s in n_games:
    if get_score_from_index(i) > min_score:
      high_scores.append((i,s))

  #Return the game tuple (game index, game distance score) and store in a dataframe
  for i,s in n_games[:how_many]:
    #Dataframe will contain attributes based on game index
    row = {'Game Title': get_title_from_index(i), 'Year': get_title_year_from_index(i), 'Score': get_score_from_index(i),
           'Weighted Score': get_weighted_score_from_index(i),
           'Total Ratings': get_total_ratings_from_index(i)}
    # print(row)
    #Append each row to this dataframe
    # recomm_df = recomm_df.append(row, ignore_index = True)
    recomm_df = pd.concat([recomm_df, pd.DataFrame([row])], ignore_index=True)
  #Sort dataframe by Sort_Option provided by user

  # recomm_df = recomm_df.sort_values(sort_option, ascending=False)
  #Only include games released same or after minimum year selected
  recomm_df = recomm_df[recomm_df['Year'] >= min_year]
  # print(recomm_df)
  return recomm_df

years_sorted = sorted(list(df['year'].unique()))


@app.route('/webhook', methods=['GET','POST'])
def webhook():
    req = request.get_json(force=True)
    # print(req)
    query_result = request.json['sessionInfo']['parameters']
    print(query_result)
    # query_result = req.get('queryResult')
    # print(query_result)
    # params = query_result['parameters']
    #
    game = query_result['game_name']
    how_many = int(query_result['no_games'])
    platform = query_result['platform']
    min_year = int(query_result['years'])
    sort_option = query_result['sort_by']
    min_score = float(query_result['min_score'])
    dropdown_option = query_result['game_name']
    output_df= gradio_contents_based_recommender_v2(game, how_many, dropdown_option, sort_option, min_year, platform, min_score)
    print(output_df.to_string())
    output_df.plot.bar(x="Game Title", y="Score", rot=0)
    # Convert the DataFrame to table data
    # table_data = dataframe_to_table(output_df)
    # webhook_response = {
    #     'fulfillment_response': {
    #         'messages': [{
    #             'text': {
    #                 'text': [output_df.to_string()]
    #             }
    #         }]
    #     }
    # }

    # Prepare the webhook response with the table
    # webhook_response = {
    #     'fulfillment_response': {
    #         'messages': [
    #             {
    #                 'payload': {
    #                     'table': {
    #                         'rows': table_data['rows'],
    #                         'columns': table_data['columns']
    #                     }
    #                 }
    #             }
    #         ]
    #     }
    # }

    fulfillment_messages = []
    if output_df.empty:
        fulfillment_message = {
            'text': {
                'text': [
                    "Sorry, I could not find any recommendations based on your preferences :(.\n Want to update your preferences ?"]
            }
        }
        fulfillment_messages.append(fulfillment_message)
    else:
        for index, row in output_df.iterrows():
            text = ''
            for column in output_df.columns:
                text += f"{column}: {row[column]}\n"
            fulfillment_message = {
                'text': {
                    'text': [text]
                }
            }
            fulfillment_messages.append(fulfillment_message)
        fulfillment_message = {
            'text': {
                'text': [
                    "Hope you like the recommendation. Would you want some more suggestion for other games ?"]
            }
        }
        fulfillment_messages.append(fulfillment_message)


    webhook_response = {
        'fulfillment_response': {
            'messages': fulfillment_messages
        }
    }

    # Send the response back to Dialogflow CX
    return jsonify(webhook_response)

def dataframe_to_table(dataframe):
    # Convert the DataFrame into table data format
    rows = dataframe.values.tolist()
    columns = [{'header': column} for column in dataframe.columns]

    table_data = {
        'rows': rows,
        'columns': columns
    }

    return table_data

if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run(host="localhost", port=8000, debug=True)
    # app.run()