## Lunit assignment

**Task**: Predict ER from image features

**Repository structure**:

- data - unpack all necessary data here (please respect original file names)
- notebooks - EDA, post-analysis, etc.
- src - source code for data preparation, training, prediction, etc.

**Env**:

- Python: 3.8
- Requirements:
  ```
  pip install -r requirements.txt --no-cache-dir
  ```

**To reproduce assignment**:

- Run an EDA [notebook](notebooks/EDA.ipynb)
- Run main pipeline (prepare/train/test)
- Run a post_analysis [notebook](notebooks/post_analysis.ipynb)

**To run pipeline, please run from project root**:

```
PYTHONPATH=. dvc repro
```
