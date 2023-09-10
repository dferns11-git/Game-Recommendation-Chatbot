# Game-Recommendation-Chatbot

Using Google Dialogflow, we built a Game Recommendation Chatbot. The chatbot prompts users to enter important information about a game they like so that it may provide recommendation based on it. 
The algorithm developed to recommend a game is based on a pivot table that is generated on the description of each game. The table developed calculates scores based on certain key words in descriptions of each game and then generates values between 0 and 1. The closer a game has a score to 1 in respect to another game, hight the probability of similarity between them. Using the number of games a user would like to get recommended as the number of games with highest similairty scores to the one the user mentioned we can generate recommendations.
