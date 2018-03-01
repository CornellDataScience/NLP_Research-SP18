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
