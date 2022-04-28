import pandas as pd
import numpy as np
from nlp_id.lemmatizer import Lemmatizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from cari_data import search_kec, search_kot
from sklearn.model_selection import KFold
from pycm import *
import re
import nltk
from nltk.tag import CRFTagger
from collections import Counter
from scipy.sparse import csr_matrix
import math
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn.model_selection import GridSearchCV

le = LabelEncoder()
lemmatizer = Lemmatizer()
ct = CRFTagger()
ct.set_model_file('/temporaryProject/Indonesian_Manually_Tagged_Corpus_ID.crf.tagger')
stpwrd = ['dan', 'ini', 'lg', 'lagi', 'lgi', 'nih', 'udah', 'dah', 'udh', 'dh', 'yg', 'yang', 'aja', 'ada', 'aja', 'ada', 'ad', 'baru', 'bisa'' bsa', 'dri', 'nya', 'juga', 'jga', 'kalo', 'kalau', 'kl', 'apa', 'itu', 'tu', 'ap', 'tuh', 'deh', 'kan', 'uda', 'kok', 'lah', 'sih', 'pas', 'mah', 'klo', 'klu', 'pula', 'iya', 'yaa', 'ya', 'kek', 'stay', 'fix', 'daan', 'huhuhu', 'haha', 'hahaha','hahha', 'hihi', 'hihhh', 'huhu', 'duh', 'wkwk', 'wkwwkkwkwk']

df = pd.read_csv('/temporaryProject/dataset_TA_fix.csv')

def postag(document):
    temp = []
    for i in document:
        temp.append(''.join(i))
    return temp
def pre_process(list_tweet):
    hasil_lemma = [lemmatizer.lemmatize(tweet) for tweet in list_tweet]
    temporary = []

    for tweet in hasil_lemma:
        wrd = []
        for word in tweet.split():
            if word not in stpwrd:
                wrd.append(word)
        temporary.append(' '.join(wrd))

    final = [' '.join(i) for i in map(postag, [ct.tag(word) for word in [nltk.word_tokenize(i) for i in temporary]])]

    return final
def model_eval():
    model_to_set = OneVsRestClassifier(SVC())

    parameters = {
        "estimator__kernel": ["rbf"],
        "estimator__C": [0.1, 0.4, 0.7, 0.9, 1, 2, 3, 4, 5, 8, 11, 14, 17, 20],
        "estimator__gamma": [0.1, 0.4, 0.7, 0.9, 1, 2, 3, 4, 5, 8, 11, 14, 17, 20],
    }

    model_tunning = GridSearchCV(model_to_set, param_grid=parameters, cv=8, return_train_score=False)
    model_tunning.fit(X, y)

    temp = pd.DataFrame(model_tunning.cv_results_)

    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 100

    temp[["param_estimator__gamma", "param_estimator__C", 'mean_test_score', 'rank_test_score']].to_csv('hasil_eval.csv')
    print(temp[["param_estimator__gamma", "param_estimator__C", 'mean_test_score', 'rank_test_score']])
def conf_matrix():
    for k in [2, 4, 6, 8]:
        print('hasil cv dengan nilai k:', k)
        kf = KFold(n_splits=k)

        eval = {
            'TP': [],
            'TN': [],
            'FP': [],
            'FN': [],
        }
        dataset = df.tweet

        clf = OneVsRestClassifier(SVC(C=0.7, kernel='rbf', gamma=2))

        for train, test in kf.split(dataset):
            X_train, X_test = dataset.iloc[train], dataset.iloc[test]
            y_train, y_test = y[train], y[test]

            X_train = [x for x in X_train]
            X_test = [x for x in X_test]

            X_train = transform(X_train)
            X_test = transform(X_test)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred)

            eval['TP'].append(cm.TP)
            eval['TN'].append(cm.TN)
            eval['FP'].append(cm.FP)
            eval['FN'].append(cm.FN)

        for k in range(0, len(le.classes_)):
            TP = np.mean([i[k] for i in eval['TP']])
            TN = np.mean([i[k] for i in eval['TN']])
            FP = np.mean([i[k] for i in eval['FP']])
            FN = np.mean([i[k] for i in eval['FN']])

            print(f'{le.classes_[k]}:\n'+"TP:", TP,
                  "\nTN:", TN,
                  "\nFP:", FP,
                  "\nFN:", FN,
                  "\nPrecision:", TP / (TP + FP),
                  "\nRecall:", TP / (TP + FN),
                  "\nF1-score:", TP / (TP + 0.5 * (FP + FN)),
                  "\nAccuracy:", (TP + TN) / (TP + TN + FP + FN)
                  )
            print()

df['tweet'] = pre_process(df.tweet)
X = [tweet for tweet in df.tweet]
y = le.fit_transform(df.bencana)

def IDF(corpus, unique_words):
    idf_dict = {}
    N=len(corpus)
    for i in unique_words:
        count = 0
        for sen in corpus:
            if i in sen.split():
                count = count+1
            idf_dict[i] = (math.log((1+N)/(count+1)))+1
    return idf_dict
def fit(whole_data):
    unique_words = set()
    if isinstance(whole_data, (list,)):
        for x in whole_data:
            for y in x.split():
                if len(y) < 2:
                    continue
                unique_words.add(y)
        unique_words = sorted(list(unique_words))
        vocab = {j: i for i, j in enumerate(unique_words)}
        Idf_values_of_all_unique_words=IDF(whole_data,unique_words)
    return vocab, Idf_values_of_all_unique_words
Vocabulary, idf_of_vocabulary=fit(X)

def POS_TF(list_sent, num_word_in_sent):
    sum_weight = []

    for i in list_sent:
        if bool(re.search(r'NN$|VB$', i)):
            sum_weight.append(num_word_in_sent*5)
        elif bool(re.search(r'JJ$|RB$', i)):
            sum_weight.append(num_word_in_sent*3)
        else:
            sum_weight.append(num_word_in_sent)

    return sum(sum_weight)
def transform(dataset, vocabulary=Vocabulary, idf_values=idf_of_vocabulary):
    sparse_matrix = csr_matrix((len(dataset), len(vocabulary)), dtype=np.float64)
    for row in range(0, len(dataset)):
        number_of_words_in_sentence = Counter(dataset[row].split())
        for word in dataset[row].split():
            if word in list(vocabulary.keys()):
                tf_idf_value=(number_of_words_in_sentence[word]/POS_TF(dataset[row].split(), number_of_words_in_sentence[word]))*(idf_values[word])
                sparse_matrix[row,vocabulary[word]]=tf_idf_value
    output = normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False)
    return output

X = transform(X)

svm = OneVsRestClassifier(SVC(kernel='rbf', C=0.7, gamma=2)).fit(X, y)

keyword = sys.argv[1] + ' ' + '-filter:retweets'

if len(sys.argv[4]) <= 0:
    hasil_pencarian = search_kot(keyword, sys.argv[2], sys.argv[3], sys.argv[5])
else:
    hasil_pencarian = search_kec(keyword, sys.argv[2], sys.argv[3], sys.argv[4])

test_set = pre_process(hasil_pencarian.tweet)
test_set = [i for i in test_set]
i = 1

for temp in zip(hasil_pencarian[['username', 'tweet', 'addressPoint']].values, transform(test_set)):
    if svm.predict(temp[1].toarray())[0] == 0:
        print('idx:'+str(i), 'usr:'+temp[0][0], 'tweet:'+temp[0][1], temp[0][2])
        i += 1
    elif svm.predict(temp[1].toarray())[0] == 1:
        print('idx:'+str(i), 'usr:'+temp[0][0], 'tweet:'+temp[0][1], temp[0][2])
        i += 1
    elif svm.predict(temp[1].toarray())[0] == 3:
        print('idx:'+str(i), 'usr:'+temp[0][0], 'tweet:'+temp[0][1], temp[0][2])
        i += 1

# conf_matrix()
# model_eval()
