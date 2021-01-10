import pandas as pd

#print(message)

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

messages= pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"])

## cleaning a data set
ps= PorterStemmer()
corpus =[]
for i in range(0, len(messages)):
    review= re.sub('[^a-zA-Z]', ' ', messages['message'][i] )
    review= review.lower()
    review= review.split()
    review= [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    print(corpus)
    exit(1)

## converting into vectors

cv= CountVectorizer(max_features=2500)
x= cv.fit_transform(corpus).toarray()
y= pd.get_dummies(messages['label'])
y=y.iloc[:,1].values
# print(y)

## training the data
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.20, random_state=0)


spam_detection=MultinomialNB().fit(x_train, y_train)

y_pred= spam_detection.predict(x_test)
# print(y_pred)

## checking accuracy
confusion_m= confusion_matrix(y_test, y_pred)
print(confusion_m)
acc= accuracy_score(y_test, y_pred)
print(acc)

