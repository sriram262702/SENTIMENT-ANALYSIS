import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Load the dataset
data = pd.read_csv("C:/Users/PC/Desktop/SENTIMENT ANALYSIS/amazonreview.csv")
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
    
    #Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    #Return the cleaned text
    return ' '.join(lemmatized_tokens)

#Apply preprocessing
data['newreviewText'] = data['reviewText'].apply(preprocesstext)

#Generate the word cloud from the preprocessed text
all_words = ' '.join(data['newreviewText'])

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

#Seeing data relations
s = data.iloc[:,4:8]
s.corr()

#Plotting the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot for each sentiment score
plt.scatter(data['Compound_score'], data['Negative_score'], color='red', label='Negative Score')
plt.scatter(data['Compound_score'], data['Neutral_score'], color='blue', label='Neutral Score')
plt.scatter(data['Compound_score'], data['Positive_score'], color='green', label='Positive Score')

#Adding labels and title
plt.title('Polarity Scores')
plt.xlabel('Compound Score')
plt.ylabel('Sentiment Scores')
plt.legend(loc='upper right')

#Show the plot
plt.show()

#Save the results to a new CSV file
data.to_csv(r"C:\Users\PC\Desktop\sentimentO.csv")
