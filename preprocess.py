# coding=utf-8
import re
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


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
