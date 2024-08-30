#Importing librarires required
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

#Load the dataset
data = pd.read_csv("C:/Users/PC/Desktop/amazonreview.csv")

#Views all the column name in the dataset
data.columns


#Function to preprocess text
def preprocesstext(text):
    
    #Convert text to lowercase
    text = text.lower()
    
    #Remove special characters, numbers, and URLs
    text = re.sub(r'[^a-z\s]', '', text)
    
    #Tokenize the text
    tokens = word_tokenize(text)
    
    #Filter out stop words
    filtered_tokens =  [token for token in tokens if token not in stopwords.words('english')]
    
    #Intializing WordNetLemmatizer function
    lemmatizer = WordNetLemmatizer()
    
    #Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    #Return the cleaned text
    return ' '.join(lemmatized_tokens)

#Apply preprocessing
data['newreviewText'] = data['reviewText'].apply(preprocesstext)

#Joining all the splitted words
all_words = ' '.join(data['newreviewText'])

#Generate the word cloud from the preprocessed text
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(all_words)

#Display the word cloud
plt.figure(figsize=(15, 5))
plt.imshow(wordcloud, interpolation='bilinear')

#No axis labels
plt.axis('off')  
plt.title("Most Common Words in Amazon Reviews")
plt.show()

#Initialize sentiment analyzer
sentianalyzer = SentimentIntensityAnalyzer()

#Function to get sentiment scores
def get_sentiment_scores(text):
    scores = sentianalyzer.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']

#Apply sentiment analysis and create new columns for the scores
data[['Negative_score', 'Neutral_score', 'Positive_score', 'Compound_score']] = data['newreviewText'].apply(lambda x: pd.Series(get_sentiment_scores(x)))

#Function to classify sentiment based on custom thresholds
def sentiment(compound):
    if compound >= -1 and compound < -0.4:
        return 'Negative'
    elif compound >= -0.4 and compound <= 0.4:
        return 'Neutral'
    else:
        return 'Positive'

#Apply the classification to create the sentiment column
data['sentiment'] = data['Compound_score'].apply(sentiment)

#Seeing data dependency pos,neu,neg vs compound
s = data.iloc[:,4:8]
s.corr()

#Selecting the maximum score among the positive,neutral and negative
data['Highest_Score'] = data[['Positive_score', 'Neutral_score', 'Negative_score']].max(axis=1)

#Selecting the highest sentiment
data['Highest_Sentiment'] = np.select(
    [
        data['Highest_Score'] == data['Positive_score'],
        data['Highest_Score'] == data['Neutral_score'],
        data['Highest_Score'] == data['Negative_score']
    ],['Positive', 'Neutral', 'Negative']
)

#Plot the scatter plot of Compound Score vs. Highest Sentiment Score with varying sizes
plt.figure(figsize=(10, 6))

#Define colors for each sentiment category
colors = {'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'}

#Define sizes based on the Highest_Score, scaling them for better visualization
sizes = data['Highest_Score'] * 200  # Scale factor to adjust the size of the points

#Scatter plot of Compound Score vs. Highest Sentiment Score with varying sizes
plt.scatter(data['Compound_score'],data['Highest_Score'],c=data['Highest_Sentiment'].map(colors), 
    s=sizes,alpha=0.6,edgecolors='black'
)

#Adding labels, title, and grid
plt.title('Compound Score vs. Highest Sentiment Score with Varying Sizes')
plt.xlabel('Compound Score')
plt.ylabel('Highest Sentiment Score')
plt.grid(True)

#Display the plot
plt.show()

#Save the results to a new CSV file
data.to_csv(r"C:\Users\PC\Desktop\sentimentO.csv")