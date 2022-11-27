# coding=utf-8
import re
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers


# df is the training dataset
# X_train: stop words will be removed from all the essays, each essay will be saved as one entry for the list
# y_train: will be returned as 6 lists, each list with length the same as the number of training data points
def prepare_dataset(df):
    STOPWORDS = set(stopwords.words('english'))

    # remove all tab char, newline char, and punctuation
    df['full_text'] = df["full_text"].replace(re.compile(r'[\n\t]'), ' ', regex=True)
    df['full_text'] = df["full_text"].replace(re.compile(r'[^\w\s]'), '', regex=True)

    # remove stopwords
    sentences_pre = df['full_text'].values
    sentences = []
    for line in sentences_pre:
        line = line.lower()
        text_tokens = word_tokenize(line)
        no_sw = [word for word in text_tokens if not word in STOPWORDS]
        filtered_sentence = (" ").join(no_sw)
        sentences.append(filtered_sentence)

    y = []
    for i in range(len(df['full_text'])):
        tmp = [df['cohesion'][i], df['syntax'][i], df['vocabulary'][i],
               df['phraseology'][i], df['grammar'][i], df['conventions'][i]]
        y.append(tmp)

    return dict(
        X_train=sentences,
        y_train=y
    )

def pre_process_v2(df_train, df_test):
    # Data pre-processing
    # remove '\n \r \w'
    df_train['full_text'] = df_train["full_text"].replace(re.compile(r'[\n\r\t]'), ' ', regex=True)
    df_test['full_text'] = df_test["full_text"].replace(re.compile(r'[\n\r\t]'), ' ', regex=True)
    df_train['full_text'] = df_train["full_text"].replace(re.compile(r'[^\w]'), ' ', regex=True)
    df_test['full_text'] = df_test["full_text"].replace(re.compile(r'[^\w]'), ' ', regex=True)

    # remove stop words
    nltk.download('stopwords')
    stop = stopwords.words('english')
    df_train['full_text'] = df_train['full_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df_test['full_text'] = df_test['full_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    print(df_train['full_text'][2])

    # count max word length
    df_train['num_words'] = df_train['full_text'].apply(lambda x: len(x.split()))
    max_words = round(df_train['num_words'].max())
    print('max word:{}'.format(max_words)) ## currently 672

    # tokenize
    nltk.download('punkt')

    # visualize words after tokenize
    train_token = df_train['full_text'].apply(word_tokenize)
    test_token = df_test['full_text'].apply(word_tokenize)

    print(train_token[0])

    # assign x and y
    X = df_train['full_text']
    y = df_train[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]

    #split train test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) 

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    print(X_train[0])
    print(X_train.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(X_val.shape)
    print(y_test.shape)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index

    train_seq = tokenizer.texts_to_sequences(X_train)
    pad_train = pad_sequences(train_seq, maxlen=max_words, truncating='post')

    val_seq = tokenizer.texts_to_sequences(X_val)
    pad_val = pad_sequences(val_seq, maxlen=max_words, truncating='post')

    test_seq = tokenizer.texts_to_sequences(X_test)
    pad_test = pad_sequences(test_seq, maxlen=max_words, truncating='post') #max length of word is 1250

    print(pad_train[0]) #pad_train is a numpy array
    print(pad_train.shape)

    word_idx_count = len(word_index)
    print(word_idx_count)

    return pad_train, pad_val, pad_test, word_idx_count





# this is to demonstrate how to use preprocess function
def main():
    df_train = pd.read_csv('./data/train.csv')

    processed = prepare_dataset(df_train)
    print("X_train length: ", len(processed['X_train']))
    print("y_train length: ", len(processed['y_train']))

    # simple vectorization mehtod
    vectorizer = TfidfVectorizer(min_df=10, lowercase=True)
    vectorizer.fit(processed['X_train'])
    X_train = vectorizer.transform(processed['X_train'])
    print(X_train[0])

    # # Uncomment if you want to visualize what X_train or y_train will look like
    # with open(r'./tmp.txt', 'w') as fp:
    #     for line in processed['X_train']:
    #         fp.write("%s\n" % line)
    #     print('Done')


if __name__ == "__main__":
    main()
