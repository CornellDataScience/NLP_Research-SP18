"""
Pre-process training data.
"""

import argparse
import json
import pandas as pd
import pickle
import os
import spacy
import sys
import time


def process(data, save=True):
    all_questions = []
    for topic in data:
        article = topic['title']
        for i, paragraph in enumerate(topic['paragraphs']):
            for qa in paragraph['qas']:
                question = qa['question']
                for ans in qa['answers']:
                    answer = ans['text']
                    answer_idx = ans['answer_start']
                    all_questions.append([article, i, question, answer, answer_idx])
    headings = ['Topic', 'Paragraph #', 'Question', 'Answer', 'Pointer']
    dataframe = pd.DataFrame(all_questions, columns=headings)
    if save:
        dataframe.to_csv('data/raw_questions.csv', index=False)
    return dataframe


def tokenize(docs):
    """Tokenize a list of documents using spacy."""
    nlp = spacy.load('en')
    t0 = time.time()
    parsed = []
    # TODO: parallelize for loop
    for i, doc in enumerate(docs):
        parsed.append(nlp(str(doc)))
        if (i + 1) % 100 == 0:
            print('Processed {} answers'.format(i + 1))
    t1 = time.time() - t0
    print('\nProcessed {} answers in {:.2f}s'.format(len(docs), t1))

    with open('data/parsed_answers.pkl', 'wb') as f:
        pickle.dump(parsed, f)


def main(args):
    with open(args.file, 'r') as file:
        data = json.load(file)['data']
    if ['raw_questions.csv'] in os.listdir('data'):
        dataframe = process(data, save=True)
    else:
        dataframe = pd.read_csv('data/raw_questions.csv')
    tokenize(dataframe.loc[:, 'Answer'].values)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='data/train-v1.1.json', help='path to SQuAD training data JSON file')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
