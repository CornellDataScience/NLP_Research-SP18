# Progress Report

02/28 (Yuji):
* Reading: [SQuAD](https://arxiv.org/pdf/1606.05250.pdf)
* Created `process.py`
  * Organize raw JSON data into tabular format. `data/raw_questions.csv`
  contains 87599 question/answer pairs, where each row records
  the article title, paragraph number, question, answer, and
  answer pointer. None of the text has been vectorized.
  * Sample usage: `python process.py data/train-v1.1.json`

02/28 (Kenta):
* Downloaded pre-trained glove to the team server
* Created `glove.py`
  * Load pre-trained model.
  * Return vectorized word if it exists in the vocabulary, `None` otherwise.
  * Sample usage:
  ```python
    from glove import Glove
    model = Glove()
    model.vectorize('hello')
  ```

03/01 (Yuji):
* Added `tokenize()` in `process.py`
  * Tokenize all answers using spaCy. This splits all answers into
  individual words (with intelligent constraints such as U.S.A. should
  not be split into three words), and also does POS tagging, lemmatization,
  dependency tree construction, etc.
  See [documentation](https://spacy.io/usage/spacy-101) for details.
  * Notes:
    * Tokenizing process should be parallelized (but isn't yet)
    * Resulting file is somewhat large (approx. 500MB)
* Some analysis of what most answers are like
  [here](https://github.com/CornellDataScience/NLP_Research-SP18/tree/master/qa_transfer/analysis/answer_distribution.ipynb).

03/02 (Kenta):
* Downloaded pre-trained Word2Vec to the team server
* Renamed `glove.py` to `pretrained_word_models.py`
  * Load pre-trained model.
  * Return vectorized word if it exists in the vocabulary, `None` otherwise.
  * Sample usage:
  ```python
    from pretrained_word_models import Pretrained_Word2Vec
    model = Pretrained_Word2Vec()
    model.vectorize('hello')
  ```
