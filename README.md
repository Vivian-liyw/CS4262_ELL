# CS4262_ELL

## preprocess.py

The preprocess.py file will clean the input files, removing all newline char, tab char, and punctuation.

It will also reformat the input to two lists X_train, y_train.

X_train is a 3911 x 1 vector, X_train[0] indicate the first input within training data.

y_train is a 3911 x 6 vector, y_train[0] contain the result of 6 metrics for the first training data, with order of [cohesion, syntax, vocabulary, phraseology, grammar, conventions].
In other words y_train[0][0] will be cohesion score for the first X_train input.

The main function here is just to demonstrate how to use the prepare_dataset() function. It also provides a simple vectorization method to play with.

## initial_analysis.ipynb

Initial_analysis.ipynb is to provide a visual understanding of the dataset that we currently have.
