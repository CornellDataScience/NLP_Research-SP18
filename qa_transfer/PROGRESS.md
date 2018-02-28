# Progress Report 

02/28 (Yuji):
* Reading: [SQuAD](https://arxiv.org/pdf/1606.05250.pdf)
* Created `process.py`
  * Organize raw JSON data into tabular format. `data/raw_questions.csv`
  contains 87599 question/answer pairs, where each row records
  the article title, paragraph number, question, answer, and 
  answer pointer. None of the text has been vectorized.
  * Sample usage: `python process.py data/train-v1.1.json`