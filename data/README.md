# Data folder

This folder stores Telugu POS-tagged data.

## Layout

- `raw/` — original datasets. The loader never overwrites an existing file here except when it creates the bundled sample corpus because the folder is empty.
- `processed/` — train/test split summaries written during training.

## If you already have a Telugu POS corpus

Copy it into `data/raw/` and keep the original elsewhere as a backup. Supported layouts:

### 1. Word and tag per line (recommended)

```
నేను	PRON
పాఠశాలకు	NOUN
వెళ్తున్నాను	VERB
.

ఆమె	PRON
ఉంది	VERB
.
```

Blank lines separate sentences. Columns may be tab-separated or space-separated. CoNLL-style `index word ... tag` lines are also accepted.

### 2. Sentence and tag sequence

```
నేను పాఠశాలకు వెళ్తున్నాను,PRON NOUN VERB
```

or tab-separated `sentence<TAB>tags`.

## Sample dataset

`sample_telugu_pos.txt` is a **demonstration corpus** generated from a small hand-built lexicon. It is large enough to train a toy BiLSTM and to demo spellchecking, but it is **not** a research Telugu treebank.

For meaningful accuracy numbers, replace it with a real corpus such as:

- a Telugu treebank / IL-ILMT POS data
- UD Telugu if available
- any in-house tagged Telugu dataset in the formats above

Do not mix fake resume numbers with results from this sample file.
