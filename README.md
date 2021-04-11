DyNet LSTM tagger based on https://github.com/yuvalpinter/Mimick

Requires Python 3 with DyNet >= 2.0. Tested on CPU only.

# Installation

```
pip3 install numpy dynet tqdm
git clone https://github.com/yvesscherrer/lstmtagger.git
```

# Basic usage

Training a model:

```
python3 model.py \
    --log-dir exp1 \
    --training-data uk_iu-ud-train.conllu \
    --dev-data uk_iu-ud-dev.conllu \
    --num-epochs 20 \
    --vocab-save exp1/vocab.pkl \
    --settings-save exp1/settings.pkl \
    --params-save exp1/params.bin
```

This command:

* Creates a directory `exp1` where log files and model files are stored
* Trains a tagger on the file `uk_iu-ud-train.conllu` (files are assumed to be in CoNLL-U format, as on [Universal Dependencies](https://universaldependencies.org) )
* Evaluates the model periodically on the file `uk_iu-ud-dev.conllu`
* Stops training after 20 epochs
* Writes the model parameters to `exp1/vocab.pkl`, `exp1/settings.pkl` and `exp1/params.bin`

A trained model can then be loaded to tag new files as follows:

```
python3 model.py \
    --log-dir exp1 \
    --vocab exp1/vocab.pkl \
    --settings exp1/settings.pkl \
    --params exp1/params.bin \
    --test-data uk_iu-ud-test.conllu \
    --test-data-out testout.txt
```

# Usage with notebook (Jupyter notebook, Google Colab)

The file `notebook.py` shows how the tagger can be called from within a notebook.

# Pretrained models

The models used in our paper *New developments in tagging pre-modern orthodox Slavic texts* (Y. Scherrer, S. Mocken & A. Rabus, Scripta & e-Scripta 18, 2018) are available for download here:

https://helsinkifi-my.sharepoint.com/:f:/g/personal/yvessche_ad_helsinki_fi/El5gN_-pQTZNrrjPaXDq82YBA3ZsWVPawD7vEH3r44wW8g?e=VFfHxh

Updated models (based on `torot-201870919` and `proiel-treebank-20180408`) are available for two formats of morphological descriptions:

* UD format: https://helsinkifi-my.sharepoint.com/:f:/g/personal/yvessche_ad_helsinki_fi/Ej2SKwJiQXVKkXS7IJ2o8YcBlPkrQSH-YYBubHrLh2rx5w?e=EPXnHv
* Original MTE-like format: https://helsinkifi-my.sharepoint.com/:f:/g/personal/yvessche_ad_helsinki_fi/EmlegdHrFctOvwSb7hi3cV8B8zdRyEJ6uHCE6tZlR-EJmQ?e=eJykw6 (In order to obtain the original tag format, the conversion script `convertOutput.py` needs to be applied to the output of the tagger.)

The `orig_punct` models differ from the `orig` models with regards to punctuation handling. In `orig`, all punctuation signs are absent, whereas in `orig_punct` they are treated as distinct tokens. The paper only describes the `orig` models.

These models can be loaded to tag new files as follows:

```
python3 model.py \
    --log-dir /path/to/logdirectory \
    --vocab /path/to/orig_punct.ud.torot_proiel.clstm/vocab.pkl \
    --settings /path/to/orig_punct.ud.torot_proiel.clstm/settings.pkl \
    --params /path/to/orig_punct.ud.torot_proiel.clstm/params*.bin \
    --test-data orig_punct.ud.test1.text \
    --test-data-out orig_punct.ud.test1.tagged
```


# Complete list of arguments

## Loading and saving data

`--training-data`	File in which the training data is stored (either as CoNLL-U text file or as saved pickle)  
`--training-data-save`	Pickle file in which the training data should be stored for future runs (if parameter is omitted, the training data is not saved as pickle)  
`--dev-data`	File in which the dev data is stored (either as CoNLL-U text file or as saved pickle)  
`--dev-data-save`	Pickle file in which the dev data should be stored for future runs (if parameter is omitted, the dev data is not saved as pickle)  
`--dev-data-out`	Text files in which to save the annotated dev files (epoch count is added in front of file extension)  
`--test-data`	File in which the test data is stored (either as CoNLL-U text file or as saved pickle)  
`--test-data-save`	Pickle file in which the test data should be stored for future runs (if parameter is omitted, the test data is not saved as pickle)  
`--test-data-out`	Text file in which to save the output of the model  
`--pretrained-embeddings`	File from which to read in pretrained word embeddings (if not supplied, will be random)
`--vocab`	Pickle file from which an existing vocabulary is loaded  
`--vocab-save`	Pickle file in which the vocabulary is saved  
`--settings`	Pickle file in which the model architecture is defined (if omitted, default settings are used)  
`--settings-save`	Pickle file to which the model architecture is saved  
`--params`	Binary file (.bin) from which the model weights are loaded  
`--params-save`	Binary files (.bin) in which the model weights are saved (epoch count is added in front of file extension)  
`--keep-every`	Keep model files and dev output every n epochs (default: 10, 0 only saves last)  
`--log-dir`	directory in which the log files are saved  

## Format of input files (default options are fine for UD-formatted files)

`--number-index`	Field in which the word numbers are stored (default: 0)  
`--w-token-index`	Field in which the tokens used for the word embeddings are stored (default: 1)  
`--c-token-index`	Field in which the tokens used for the character embeddings are stored (default: 1)  
`--pos-index`	Field in which the main POS is stored (default, UD tags: 3) (original non-UD tag: 4)  
`--morph-index`	Field in which the morphology tags are stored (default: 5); use negative value if morphosyntactic tags should not be considered  
`--no-eval-feats`	(Comma-separated) list of morphological features that should be ignored during evaluation; typically used for additional tasks in multitask settings  

## Options for easily reducing the size of the training data

`--training-sentence-size`	Instance count of training set (default: unlimited)  
`--debug`	Debug mode (reduces number of training and testing examples)  

## Model settings

`--num-epochs`	Number of full passes through training set (default: 40; disable training by setting --num-epochs to negative value)  
`--char-num-layers`	Number of character LSTM layers (default: 2, use 0 to disable character LSTM)  
`--char-emb-dim`	Size of character embedding layer (default: 128)  
`--char-hidden-dim`	Size of character LSTM hidden layers (default: 256)  
`--word-emb-dim`	Size of word embedding layer (ignored if pre-trained word embeddings are loaded, use 0 to disable word embeddings)  
`--fix-embeddings`	Do not update word embeddings during training (default: off, only makes sense with pretrained embeddings)
`--tag-num-layers`	Number of tagger LSTM layers (default: 2)  
`--tag-hidden-dim`	Size of tagger LSTM hidden layers (default: 256)  
`--learning-rate`	Initial learning rate (default: 0.01)  
`--decay`	Learning rate decay (default: 0.1, 0 to turn off)  
`--dropout`	Amount of dropout to apply to LSTM parts of graph (default: 0.02, -1 to turn off)  
`--loss-prop`	Proportional loss magnitudes  
