"""
Pre-process training data.
"""

import argparse
import json
import pandas as pd
import sys


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


def main(args):
    with open(args.file, 'r') as file:
        data = json.load(file)['data']
    dataframe = process(data, save=True)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='data/train-v1.1.json',
                        help='path to SQuAD training data JSON file')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))