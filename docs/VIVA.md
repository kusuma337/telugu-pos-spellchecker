# Viva preparation — Telugu POS Tagging and Spellchecking

Use these short answers in a B.Tech viva. Speak them in your own words.

## What is POS tagging?

Part-of-Speech tagging assigns a grammatical label to each word, such as noun, verb, adjective, or pronoun. Example: in “నేను పాఠశాలకు వెళ్తున్నాను”, నేను is a pronoun, పాఠశాలకు is a noun, and వెళ్తున్నాను is a verb.

## Why is POS tagging important?

Many NLP tasks need word-class information: named-entity recognition, parsing, machine translation, information extraction, and grammar checking. If the tagger is wrong, later components usually become worse.

## Why Telugu?

Telugu is a morphologically rich Dravidian language. Words change form with tense, case, and agreement. Resources are smaller than for English, so spelling errors and unknown words are common. That makes POS tagging and spellchecking a useful academic problem.

## What is spellchecking?

Spellchecking finds words that are unlikely in the language and suggests a closer valid word. In this project it runs **before** POS tagging so the tagger sees cleaner input.

## Why probabilistic spellchecking?

A dictionary-only checker says “unknown / known”. A probabilistic checker ranks candidates using:

`P(correct | misspelled) ∝ P(misspelled | correct) × P(correct)`

- `P(correct)` comes from word frequency in the training corpus.
- `P(misspelled | correct)` is modeled with edit distance (insert, delete, replace, transpose).

This is a noisy-channel model, similar to classic work by Kernighan, Church, and Gale, and to Norvig-style correctors.

## What is an embedding?

An embedding maps a word ID to a dense vector. Similar words can get similar vectors. The BiLSTM reads these vectors instead of raw text.

## What is an LSTM?

A Long Short-Term Memory network is a recurrent model that keeps a memory cell and gates. It can use earlier words in the sentence when tagging the current word, and it is more stable than a plain RNN on longer sequences.

## What is a bidirectional LSTM?

A BiLSTM runs one LSTM left-to-right and another right-to-left, then concatenates the hidden states. For POS tagging, both previous and following words help (for example, a word before a verb is often a noun or pronoun).

## Why BiLSTM instead of a normal LSTM?

A one-direction LSTM only sees the past. POS often depends on the next word as well. Bidirectional context usually improves tagging, especially for ambiguous words.

## What is BiLSTM in this project?

Architecture:

1. Word IDs
2. Embedding layer
3. Bidirectional LSTM
4. Dropout
5. Linear layer
6. Softmax / argmax over POS tags

Loss is cross-entropy. Padding tokens are ignored so they do not affect training.

## What is CRF?

A Conditional Random Field is a sequence model that scores a whole tag path, not each token independently. It can learn that some tag pairs are rare (for example, two consecutive punctuation tags). This project compares a neural tagger with a simple most-frequent-tag baseline. A CRF was not added as a hard dependency because popular CRF libraries are painful on Windows. You can mention CRF as related work or as a future baseline.

## What is F1-score?

F1 is the harmonic mean of precision and recall:

`F1 = 2 × (precision × recall) / (precision + recall)`

Macro F1 averages F1 across POS classes, so rare tags still matter.

## Why accuracy can be misleading?

If most words are nouns, a model that always predicts NOUN can look accurate but fail on verbs and pronouns. Accuracy also ignores class imbalance. F1, precision, and recall give a clearer picture.

## How does spellchecking improve POS tagging?

The tagger was trained on correctly spelled corpus words. A misspelling often becomes `<UNK>` and is tagged poorly. If the spellchecker restores a known word, the embedding and LSTM see a familiar token and tagging improves.

## What happens when a word is unknown?

Unknown words map to the `<UNK>` embedding. The BiLSTM can still use neighboring words. The spellchecker tries nearby vocabulary words first; if no candidate beats the threshold, the original token is kept.

## What is the input and output of the model?

- Input: a Telugu sentence, tokenized into words.
- Spellchecker output: a possibly corrected token list.
- Tagger input: integer word IDs, packed by sentence length.
- Tagger output: a POS tag for each token.

## What are the limitations?

- The bundled dataset is a small demonstration corpus, not a treebank.
- The error model uses edit distance, not real Telugu typing patterns.
- No character-level or morphological features.
- Context-independent spellchecking (each word is scored alone).
- Reported literature numbers such as 71.7% / 75.8% are **not** copied here; this repository reports only scores produced by training on the data you actually have.

## What are future improvements?

- Train on a real Telugu POS corpus.
- Add character CNN/LSTM features for unknown words.
- Use a CRF layer on top of BiLSTM.
- Contextual language-model spellchecking.
- Handle sandhi and rich morphology more carefully.
- Cross-validation and error analysis by tag type.
