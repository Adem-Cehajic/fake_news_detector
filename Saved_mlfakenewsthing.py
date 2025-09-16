import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
swords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
vectorizer=TfidfVectorizer()
#createfile
'''
FTRAIN_PATH = "./Data/False.csv"
TTRAIN_PATH = "./Data/Truth.csv"
y = pd.read_csv(TTRAIN_PATH)
n = pd.read_csv(FTRAIN_PATH)
y['Label'] = 0
n['Label'] = 1
y.to_csv('./Data/TrainingProcessedData.csv',index=False)
n.to_csv('./Data/TrainingProcessedData.csv',index=False,mode='a',header=False)
'''
training = './Data/TrainingProcessedData.csv'
traindf = pd.read_csv(training)
def stemmer(text):
    #function removes non word characters and makes a string lowercase inorder to remove stop words and use a lemmatizer algorithm
    text = str(text)
    ptext = re.sub(r'[^\w\s]', '', text)
    ptext = ptext.lower()
    ptext = ptext.split()
    ptext = [lemmatizer.lemmatize(word) for word in ptext if word not in swords]
    ptext = ' '.join(ptext)
    return ptext
#lemmatize and remove unnecessary columns
traindf['title'] = traindf['title'].apply(stemmer)
traindf['text'] = traindf['text'].apply(stemmer)
traindf['subject'] = traindf['subject'].apply(stemmer)
traindf.drop(columns=('date'),inplace=True)
y = traindf['Label'].values
#vectorize data
titlevector = vectorizer.fit_transform(traindf['title'])
textvector = vectorizer.fit_transform(traindf['text'])
subjectvector = vectorizer.fit_transform(traindf['subject'])
x = hstack([titlevector,textvector,subjectvector])
#creating logistical regression
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
lr = LogisticRegression()
lr.fit(X_train,y_train)
ypred = lr.predict(X_test)
#print(classification_report(y_test,ypred))
#with open('linear_regression_model.pkl', 'wb') as file:
    #pickle.dump(lr, file)
with open('linear_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print(loaded_model.predict(X_test[110]))