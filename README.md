# Telugu POS Tagging and Intelligent Spellchecking

A Telugu NLP application that combines a **Bidirectional LSTM (BiLSTM) Part-of-Speech tagger** with a **probabilistic spellchecker**. The project provides an interactive Streamlit interface for sentence analysis, spelling suggestions, POS predictions, model comparison, batch processing, and document review.

## Key Features

- **Telugu POS Tagging:** Predicts grammatical tags for tokens in Telugu text.
- **Probabilistic Spellchecking:** Suggests spelling corrections using edit distance and unigram probabilities.
- **Token Confidence Scores:** Displays model confidence scores for POS predictions.
- **Sentence Analysis:** Analyzes Telugu text using the trained NLP pipeline.
- **POS Analysis:** Explores predicted grammatical categories and text statistics.
- **Model Comparison:** Compares baseline, BiLSTM, noisy-input, and combined pipeline results.
- **Batch Processing:** Supports text-file and CSV workflows in the existing interface.
- **Telugu Document Doctor:** Reviews uploaded UTF-8 text documents sentence by sentence, presents spelling suggestions and POS predictions, and lets users choose which corrections to accept.
- **Downloadable Reports:** Exports reviewed text and JSON analysis reports from Document Doctor.
- **Automated Tests:** Includes a test suite for analytics, preprocessing, model, spellchecker, and data loading.

> Spellchecking suggestions and POS predictions may be incorrect. Review suggested corrections before using them in important documents.

## Model Performance

The following results were obtained from the project's evaluation command:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Most Frequent Tag baseline | 95.54% | 75.55% | 73.85% | 0.7425 |
| BiLSTM | 96.43% | 83.48% | 76.41% | 0.7823 |
| BiLSTM on noisy input | 90.18% | 78.42% | 71.09% | 0.7119 |
| BiLSTM + Spellcheck | **97.32%** | 82.59% | 78.97% | **0.8003** |

The reported evaluation used a held-out 15% test split containing 112 tokens across 23 sentences. These results describe this evaluation split and should not be interpreted as a guarantee of performance on all Telugu text.

## Architecture

```text
Telugu Text
    |
    v
Tokenization and Unicode Normalization
    |
    v
Probabilistic Spellchecker
    |
    v
BiLSTM POS Tagger
    |
    v
POS Tags and Confidence Scores
    |
    v
Streamlit Interface and Document Review
```

### Model Details

- Embedding dimension: 128
- Bidirectional LSTM hidden dimension: 128 per direction
- Number of LSTM layers: 1
- Dropout: 0.3
- Maximum sequence length: 64
- Batch size: 8
- Training epochs: 25, with early stopping configured
- Optimizer: Adam

## Technology Stack

- Python 3.11
- PyTorch
- Streamlit
- pandas
- scikit-learn
- NLP preprocessing and probabilistic spellchecking
- pytest

## Project Structure

```text
telugu-pos-spellchecker/
├── app/
│   ├── streamlit_app.py
│   ├── analytics.py
│   └── document_doctor.py
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── vocabulary.py
│   ├── pos_model.py
│   ├── spellchecker.py
│   ├── inference.py
│   ├── data_loader.py
│   ├── train.py
│   └── evaluate.py
├── tests/
├── data/
├── models/
├── results/
├── requirements.txt
├── pytest.ini
└── README.md
```

Some generated model artifacts and datasets may be excluded from version control. Check the repository files before sharing the project so that the required inference artifacts are available to anyone running the application.

## Setup and Installation

Run these commands from the project root.

### 1. Create and activate a virtual environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

If `requirements.txt` is present and up to date:

```cmd
python -m pip install -r requirements.txt
```

### 3. Run the application

```cmd
streamlit run app/streamlit_app.py
```

Open the local URL displayed in the terminal.

### 4. Evaluate the model

```cmd
python -m src.evaluate
```

### 5. Run automated tests

```cmd
python -m pytest -v
```

**Latest verified test result:** 64 tests passed. Run the test command again after making changes to verify the current checkout.

### 6. Train the model

Training is optional when the required trained artifacts are already available. To train using the project's configured dataset:

```cmd
python -m src.train
```

Make sure the expected training data and configuration are available before retraining.

## Using Telugu Document Doctor

1. Start the Streamlit application.
2. Open the **Document Doctor** tab.
3. Upload a UTF-8 encoded `.txt` document.
4. Select **Analyze Document**.
5. Review sentence-level spelling suggestions and POS predictions.
6. Select the corrections you want to accept.
7. Download the reviewed text or JSON analysis report.

The current document workflow supports text files. It is not a complete Telugu grammar-correction system.

## Testing

The test suite covers project components such as analytics, data loading, model behavior, preprocessing, and spellchecking. The current verified suite contains 64 passing tests.

## Future Improvements

- Add a feedback dataset from accepted and rejected spelling suggestions.
- Evaluate on a larger and more diverse Telugu test corpus.
- Improve sentence splitting and document formatting preservation.
- Add more robust evaluation of spelling correction quality.
- Publish reproducible setup instructions and application screenshots.

## Project Goal

The goal is to make Telugu text analysis more accessible by combining POS tagging, spelling suggestions, and human-reviewed document correction in one interactive application.