"""ANLP coursework 2: Executable module for span labeling on NLU++"""

import argparse
import json
import math
import os
from collections import defaultdict

import sklearn.linear_model
import sklearn.preprocessing
import spacy
import numpy as np

import utils

# Default data files
DATA_DIR = "data"
TRAINING_FILE = "training_data.json"
VALIDATION_FILE = "validation_data.json"

spacy.tokens.token.Token.set_extension("bio_slot_label", default="O")


def _predict_tag_mle(token, model_parameters):
    """Predict most-frequent-tag baseline.

    Args:
        token (spacy.tokens.Token): token whose tag we want to predict
        model_parameters (dict): a mapping token from most frequent tag, and
          an unknown word tag

    Returns:
        (List(Tuple(str,float))): a list of (tag, logprob) pairs, sorted from
          most to least probable
    """
    if token.text in model_parameters['per_word_tag_mle']:
        return model_parameters['per_word_tag_mle'][token.text]
    return model_parameters['unknown_word_distribution']


def train_tag_mle(data):
    """Train a model p(<tag> | <word>).

    Args:
        data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        Callable: a function from a word to a sorted distribution over tags.
    """
    token_count = 0

    # Count the number of times each word/tag combination appears
    # and store them so that word_tag_count[word][tag] is the count
    word_tag_count = defaultdict(lambda: defaultdict(int))
    for sample in data:
        for token in sample['annotated_text']:
            word_tag_count[token.text][token._.bio_slot_label] += 1
            token_count += 1

    # For each word, normalise the associated tag counts to obtain
    # maximum likelihood estimate of p(<tag> | <word>), and store the
    # logarithm of the probability. For each word, sort its possible tags
    # by their logprobs
    per_word_tag_mle = {}
    for word, tag_count_pairs in word_tag_count.items():
        word_count = sum(tag_count_pairs.values())
        tag_logprob_pairs = [(tag, math.log(tag_word_count / word_count)) for
                             (tag, tag_word_count) in tag_count_pairs.items()]
        sorted_tag_logprob_pairs = \
            sorted(tag_logprob_pairs,
                   key=lambda tag_logprob_pair: -tag_logprob_pair[1])
        per_word_tag_mle[word] = sorted_tag_logprob_pairs

    # The model maps each word to its sorted table of logprobs. Return a
    # function that takes a word as input, and returns this sorted table.
    # If the word is unknown, return a distribution where p('O'|<word>) == 1
    model_parameters = {
        'per_word_tag_mle': per_word_tag_mle,
        'unknown_word_distribution': [('O', 1)]
    }

    print(f'> Trained most frequent tag model on {token_count} tokens')
    return lambda token: _predict_tag_mle(token, model_parameters)


def _predict_tag_logreg(token, model, tag_encoder):
    """Predict tag according to logistic regression on word embedding.

    Args:
        token: a spacy token to tag
        model: an sklearn model that predicts tags from token vectors
        tags: a mapping from tag ids to BIO tag strings

    Returns:
        (List(Tuple(str,float))): distribution over tags, sorted from most to
          least probable
    """
    log_probs = model.predict_log_proba([token.vector])[0]
    distribution = list(zip(tag_encoder.classes_, log_probs))
    sorted_distribution = \
        sorted(distribution, key=lambda tag_logprob_pair: -tag_logprob_pair[1])
    return sorted_distribution


def train_tag_logreg(data):
    """Train a logistic regression model for p(<tag> | <word embedding>).

    Args:
        data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        Callable: a function that returns a (sorted) distribution over tags,
          given a word.
    """

    train_X = [token.vector for
               sample in data for
               token in sample['annotated_text']]
    train_y = [token._.bio_slot_label for
               sample in data for
               token in sample['annotated_text']]

    # train_X is already numeric, but train_y is strings. We must encode train_y
    # numerically in order to train the sklearn logistic regression model.
    tag_encoder = sklearn.preprocessing.LabelEncoder()
    train_y_encoded = tag_encoder.fit_transform(train_y)

    model = sklearn.linear_model.LogisticRegression(multi_class='multinomial',
                                                    solver='newton-cg' \
                                                   ).fit(train_X,
                                                         train_y_encoded)

    print(f'> Finished training logistic regression on {len(train_X)} tokens')
    return lambda token: _predict_tag_logreg(token, model, tag_encoder)


def combine_features(token, pos_encoder, dep_encoder, lemma_encoder, morph_encoder):
    features = []

    # 当前词的词嵌入
    features.extend(token.vector)

    # POS、DEP、Lemma 编码
    pos_encoded = pos_encoder.transform([[token.pos_]])[0]
    dep_encoded = dep_encoder.transform([[token.dep_]])[0]
    lemma_encoded = lemma_encoder.transform([[token.lemma_]])[0]
    features.extend(pos_encoded)
    features.extend(dep_encoded)
    features.extend(lemma_encoded)

    # 是否为数词
    features.append(int(token.pos_ == 'NUM'))

    # 词长
    features.append(len(token.text))

    # 形态学特征编码
    morph_features = token.morph.to_dict()
    morph_feature_values = [f"{key}={value}" for key, value in morph_features.items()]
    if not morph_feature_values:
        morph_feature_values = ['None']
    morph_encoded = morph_encoder.transform([morph_feature_values])[0]
    features.extend(morph_encoded)

    if token.i > 0:
        prev_token = token.doc[token.i - 1]
        features.extend(prev_token.vector)
        pos_encoded = pos_encoder.transform([[prev_token.pos_]])[0]
        dep_encoded = dep_encoder.transform([[prev_token.dep_]])[0]
        lemma_encoded = lemma_encoder.transform([[prev_token.lemma_]])[0]
        features.extend(pos_encoded)
        features.extend(dep_encoded)
        features.extend(lemma_encoded)
        features.append(int(prev_token.pos_ == 'NUM'))
    else:
        features.extend(np.zeros_like(token.vector))
        features.extend(np.zeros(pos_encoder.categories_[0].shape[0]))
        features.extend(np.zeros(dep_encoder.categories_[0].shape[0]))
        features.extend(np.zeros(lemma_encoder.categories_[0].shape[0]))
        features.append(0)

    if token.i < len(token.doc) - 1:
        next_token = token.doc[token.i + 1]
        features.extend(next_token.vector)
        pos_encoded = pos_encoder.transform([[next_token.pos_]])[0]
        dep_encoded = dep_encoder.transform([[next_token.dep_]])[0]
        lemma_encoded = lemma_encoder.transform([[next_token.lemma_]])[0]
        features.extend(pos_encoded)
        features.extend(dep_encoded)
        features.extend(lemma_encoded)
        features.append(int(next_token.pos_ == 'NUM'))
    else:
        features.extend(np.zeros_like(token.vector))
        features.extend(np.zeros(pos_encoder.categories_[0].shape[0]))
        features.extend(np.zeros(dep_encoder.categories_[0].shape[0]))
        features.extend(np.zeros(lemma_encoder.categories_[0].shape[0]))
        features.append(0)

    # Head token
    head_token = token.head
    if head_token is not token:
        features.extend(head_token.vector)
        pos_encoded = pos_encoder.transform([[head_token.pos_]])[0]
        dep_encoded = dep_encoder.transform([[head_token.dep_]])[0]
        lemma_encoded = lemma_encoder.transform([[head_token.lemma_]])[0]
        features.extend(pos_encoded)
        features.extend(dep_encoded)
        features.extend(lemma_encoded)
        features.append(int(head_token.pos_ == 'NUM'))
    else:
        features.extend(np.zeros_like(token.vector))
        features.extend(np.zeros(pos_encoder.categories_[0].shape[0]))
        features.extend(np.zeros(dep_encoder.categories_[0].shape[0]))
        features.extend(np.zeros(lemma_encoder.categories_[0].shape[0]))
        features.append(0)

    return features


def _predict_tag_mymodel(token, model_parameters):
    pos_encoder = model_parameters['pos_encoder']
    dep_encoder = model_parameters['dep_encoder']
    tag_encoder = model_parameters['tag_encoder']
    lemma_encoder = model_parameters['lemma_encoder']
    morph_encoder = model_parameters['morph_encoder']
    scaler = model_parameters['scaler']
    model = model_parameters['model']

    features = combine_features(token, pos_encoder, dep_encoder, lemma_encoder, morph_encoder)
    X = np.array(features).reshape(1, -1)
    # X = scaler.transform(X)

    log_probs = model.predict_log_proba(X)[0]
    distribution = list(zip(tag_encoder.classes_, log_probs))
    sorted_distribution = sorted(distribution, key=lambda tag_logprob_pair: -tag_logprob_pair[1])
    return sorted_distribution

def train_my_model(data):
    """Train a logistic regression model for p(<tag> | <word embedding>).

    Args:
        data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        Callable: a function that returns a (sorted) distribution over tags,
          given a word.
    """
    ##########################################################################
    #
    # Your model code goes here.
    #
    # You may find it easiest to base your code on train_tag_logreg, and
    # _predict_tag_logreg, both above. You should make sure that you understand
    # every line in the code before attempting to extend it. If you aren't
    # sure what somethings does, refer to the scikit-learn documentation:
    # https://scikit-learn.org/
    #
    # In that code, notice that string values for the output tag are encoded
    # numerically. If you use string values as input features (e.g. linguistic
    # annotations from spacy), you must decide how best to encode them
    # numerically as well. There are multiple ways to do this, see:
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
    #
    # Be sure to describe and justify all decisions in your report.
    #
    ##########################################################################
    train_X = []
    train_y = []
    pos_list = []
    dep_list = []
    lemma_list = []
    morph_feature_list = []
    is_num_list = []

    for sample in data:
        for token in sample['annotated_text']:
            pos_list.append(token.pos_)
            dep_list.append(token.dep_)
            lemma_list.append(token.lemma_)
            is_num_list.append(token.pos_ == 'NUM')

            # head token
            pos_list.append(token.head.pos_)
            dep_list.append(token.head.dep_)
            lemma_list.append(token.head.lemma_)
            is_num_list.append(token.head.pos_ == 'NUM')

            morph_features = token.morph.to_dict()
            for key, value in morph_features.items():
                morph_feature_list.append(f"{key}={value}")

    pos_encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    dep_encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    lemma_encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    morph_encoder = sklearn.preprocessing.MultiLabelBinarizer()

    pos_encoder.fit(np.array(pos_list).reshape(-1, 1))
    dep_encoder.fit(np.array(dep_list).reshape(-1, 1))
    lemma_encoder.fit(np.array(lemma_list).reshape(-1, 1))
    morph_encoder.fit(np.array(morph_feature_list).reshape(-1, 1))

    for sample in data:
        for token in sample['annotated_text']:
            features = combine_features(token, pos_encoder, dep_encoder, lemma_encoder, morph_encoder)
            train_X.append(features)
            train_y.append(token._.bio_slot_label)

    train_X = np.array(train_X)
    scaler = sklearn.preprocessing.StandardScaler()
    # train_X = scaler.fit_transform(train_X)

    tag_encoder = sklearn.preprocessing.LabelEncoder()
    train_y_encoded = tag_encoder.fit_transform(train_y)

    model = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=2000, class_weight='balanced').fit(train_X, train_y_encoded)

    model_parameters = {
        'pos_encoder': pos_encoder,
        'dep_encoder': dep_encoder,
        'tag_encoder': tag_encoder,
        'lemma_encoder': lemma_encoder,
        'morph_encoder': morph_encoder,
        'scaler': scaler,
        'model': model
    }

    print(f'> Finished training logistic regression on {len(train_X)} tokens')
    return lambda token: _predict_tag_mymodel(token, model_parameters)


def predict_independent_tags(tag_predictor, data):
    """Predict tags independently of each other.

    Args:
        tag_predictor: function that returns a sorted distribution over tags,
          given a spacy token
        data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        List[List[Str]]: a nested array in the same shape as data, where each
        element is the predicted tag
    """
    predictions = []
    for sample in data:
        predictions.append([tag_predictor(token)[0][0] for
                            token in sample['annotated_text']])
    return predictions


def predict_bio_tags(tag_predictor, data):
    """Predict tags respecting BIO tagging constraints.

    An I-<tag> can only follow I-<tag> or B-<tag>, for any <tag>.

    Args:
        tag_predictor: function that returns a sorted distribution over tags,
          given a spacy token
       data: data in NLU++ format imported from JSON. Each data sample
          should include spacy annotations in the variable 'annotated_text',
          and each token in the annotated text should include a 'bio_slot_label'.

    Returns:
        List[List[Str]]: a nested array in the same shape as data, where each
        element is the predicted tag
    """
    predictions = []
    for sample in data:
        prev_tag = 'O'
        predictions.append([])
        for token in sample['annotated_text']:
            label = 'O'
            for tag, _ in tag_predictor(token):
                if tag.startswith('I') and tag[1:] != prev_tag[1:]:
                    continue
                label = tag
                break
            predictions[-1].append(label)
            prev_tag = predictions[-1][-1]
    return predictions


def main():
    """Main program loop. Trains model and evaluates on a test set.
    """

    # Map command-line options to callable functions
    train = {
        'most_frequent_tag' : train_tag_mle,
        'logistic_regression' : train_tag_logreg,
        'my_model': train_my_model
    }
    default_train = 'most_frequent_tag'

    predict = {
        'independent_tags' : predict_independent_tags,
        'bio_tags' : predict_bio_tags,
    }
    default_predict = 'independent_tags'

    default_annotations = 'lemma_:15,tag_:8,dep_:8'

    # Read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_path',
                        help='path to training dataset in JSON format ' + \
                             f'(default={DATA_DIR}/{TRAINING_FILE})',
                        default=os.path.join(DATA_DIR, TRAINING_FILE))
    parser.add_argument('-v', '--validation_path',
                        help='path to validation dataset in JSON format ' + \
                             f'(default={DATA_DIR}/{VALIDATION_FILE})',
                        default=os.path.join(DATA_DIR, VALIDATION_FILE))
    parser.add_argument('-m', '--model',
                        help='type of model to use ' + \
                             f'(default={default_train})',
                        choices=train.keys(),
                        default=default_train)
    parser.add_argument('-p', '--predictor',
                        help='type of prediction method to use ' + \
                             f'(default={default_predict})',
                        choices=predict.keys(),
                        default=default_predict)
    parser.add_argument('-f', '--full_report',
                        help='show tags and annotations for all test tokens,',
                        action='store_true')
    parser.add_argument('-a', '--annotations',
                        help='spacy annotations to show for each word in ' + \
                             'a full report, given as ' + \
                             'comma-separated "<annotation>:<width>" ' + \
                             'pairs without spaces ' + \
                             f'(default={default_annotations})',
                        default=default_annotations)
    args = parser.parse_args()
    
    attributes = []
    col_widths = []
    for attr_col_pair in args.annotations.split(','):
        (attr, col_width) = attr_col_pair.split(':')
        attributes.append(attr)
        col_widths.append(int(col_width))

    print(f'> Reading training data from {args.train_path}...')
    with open(args.train_path) as input_file:
        training_data = json.load(input_file)

    print(f'> Reading validation data from {args.validation_path}...')
    with open(args.validation_path) as input_file:
        validation_data = json.load(input_file)

    print('> Tokenising and annotating raw data with spacy...')
    nlp_analyser = spacy.load("en_core_web_sm")
    utils.tokenise_annotate_and_convert_slot_labels_to_bio_tags(training_data,
                                                                nlp_analyser)
    utils.tokenise_annotate_and_convert_slot_labels_to_bio_tags(validation_data,
                                                                nlp_analyser)

    print(f'> Training {args.model} model on training data...')
    model = train[args.model](training_data)

    print(f'> Predicting tags on validation data...')
    predictions = predict[args.predictor](model, validation_data)

    print(f'> Evaluating results on validation data...')
    if args.full_report:
        utils.visualise_bio_tags(predictions, validation_data,
                                 attributes, col_widths)
    utils.evaluate(predictions, validation_data)


if __name__ == "__main__":
    main()
