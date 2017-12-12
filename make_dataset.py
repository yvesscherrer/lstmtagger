"""
Reads in CONLL files to make the dataset
Output a textual vocabulary file, and a pickle file of a dict with the following elements
training_instances: List of (sentence, tags) for training data
dev_instances
w2i: Dict mapping words to indices
t2is: Dict mapping attribute types (POS / morpho) to dicts from tags to indices
c2i: Dict mapping characters to indices
"""

import argparse
import pickle
import collections
import random
import utils
import sys

__author__ = "Yuval Pinter and Robert Guthrie, 2017 + Yves Scherrer"

Instance = collections.namedtuple("Instance", ["w_sentence", "c_sentence", "tags"])

UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
UNK_CHAR_TAG = "<?>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"

def read_file(filename, w2i, t2is, c2i, number_index=0, token_index=1, pos_index=3, morph_index=5, update_vocab=True):
	"""
	Read in a dataset and turn it into a list of instances.
	Modifies the w2i, t2is and c2i dicts, adding new words/attributes/tags/chars
	as it sees them.
	token_index: field in which the token representation is stored
	pos_index: field in which the main POS is stored
	morph_index: field in which the morphological tags are stored (-1 if no morphology)
	"""

	# populate mandatory t2i tables
	if POS_KEY not in t2is and update_vocab:
		t2is[POS_KEY] = {NONE_TAG: 0}

	# build dataset
	instances = []
	vocab_counter = collections.Counter()
	with open(filename, "r", encoding="utf-8") as f:

		# running sentence buffers (lines are tokens)
		w_sentence = []
		c_sentence = []
		tags = collections.defaultdict(list)

		# main file reading loop
		for line in f:

			# discard comments
			if line.startswith("#"):
				continue

			# parse sentence end
			elif line.isspace():

				# pad tag lists to sentence end
				slen = len(w_sentence)
				for seq in tags.values():
					if len(seq) < slen:
						seq.extend([0] * (slen - len(seq))) # 0 guaranteed below to represent NONE_TAG

				# add sentence to dataset
				instances.append(Instance(w_sentence, c_sentence, tags))
				w_sentence = []
				c_sentence = []
				tags = collections.defaultdict(list)

			else:

				# parse token information in line
				data = line.split("\t")
				if '-' in data[token_index]: # Some UD languages have contractions on a separate line, we don't want to include them also
					continue
				idx = int(data[number_index])
				word = data[token_index]
				postag = data[pos_index] if pos_index != morph_index else data[pos_index].split("|")[0]
				morphotags = {} if morph_index < 0 else utils.split_tagstring(data[morph_index], uni_key=False, has_pos=(pos_index == morph_index))

				# ensure counts and dictionary population
				vocab_counter[word] += 1

				if update_vocab:
					if word not in w2i:
						w2i[word] = len(w2i)
					pt2i = t2is[POS_KEY]
					if postag not in pt2i:
						pt2i[postag] = len(pt2i)
					for c in word:
						if c not in c2i:
							c2i[c] = len(c2i)
					for key, val in morphotags.items():
						if key not in t2is:
							t2is[key] = {NONE_TAG:0}
						mt2i = t2is[key]
						if val not in mt2i:
							mt2i[val] = len(mt2i)

				# add data to sentence buffer
				w_sentence.append(w2i.get(word, w2i[UNK_TAG]))
				c_sentence.append([c2i[PADDING_CHAR]] + [c2i.get(c, c2i[UNK_CHAR_TAG]) for c in word] + [c2i[PADDING_CHAR]])
				tags[POS_KEY].append(t2is[POS_KEY].get(postag, t2is[POS_KEY][NONE_TAG]))
				for k,v in morphotags.items():
					mtags = tags[k]
					# pad backwards to latest seen
					missing_tags = idx - len(mtags) - 1
					mtags.extend([0] * missing_tags) # 0 guaranteed above to represent NONE_TAG
					mtags.append(t2is[k][v])

	return instances, vocab_counter

if __name__ == "__main__":

	# parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--training-data", required=True, dest="training_data", help="Training data .txt file")
	parser.add_argument("--dev-data", required=True, dest="dev_data", help="Development data .txt file")
	parser.add_argument("--test-data", required=True, dest="test_data", help="Test data .txt file")
	parser.add_argument("--number-index", dest="number_index", type=int, help="Field in which the word numbers are stored (default: 0)", default=0)
	parser.add_argument("--token-index", dest="token_index", type=int, help="Field in which the tokens are stored (default: 1)", default=1)
	parser.add_argument("--pos-index", dest="pos_index", type=int, help="Field in which the main POS is stored (default, UD tags: 3) (original non-UD tag: 4)", default=3)
	parser.add_argument("--morph-index", dest="morph_index", type=int, help="Field in which the morphology tags are stored (default: 5); use negative value if morphosyntactic tags should not be considered", default=5)
	parser.add_argument("-o", required=True, dest="output", help="Output file id (*.[train|dev|test].pkl)")
	parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt", help="Text file containing all of the words in the train/dev/test data to use in outputting embeddings")
	parser.add_argument("--training-sentence-size", default=sys.maxsize, dest="training_sentence_size", type=int, help="Instance count of training set (default - unlimited)")
	parser.add_argument("--token-size", default=sys.maxsize, dest="token_size", type=int, help="Token count of training set (default - unlimited)")
	options = parser.parse_args()

	w2i = {UNK_TAG: 0} # mapping from word to index
	t2is = {} # mapping from attribute name to mapping from tag to index
	c2i = {UNK_CHAR_TAG: 0, PADDING_CHAR: 1} # mapping from character to index, for char-RNN concatenations

	# read data from UD files
	training_instances, training_vocab = read_file(options.training_data, w2i, t2is, c2i, options.number_index, options.token_index, options.pos_index, options.morph_index, update_vocab=True)
	dev_instances, dev_vocab = read_file(options.dev_data, w2i, t2is, c2i, options.number_index, options.token_index, options.pos_index, options.morph_index, update_vocab=False)
	test_instances, test_vocab = read_file(options.test_data, w2i, t2is, c2i, options.number_index, options.token_index, options.pos_index, options.morph_index, update_vocab=False)
	
	# trim training set for size evaluation (sentence based)
	if len(training_instances) > options.training_sentence_size:
		random.shuffle(training_instances)
		training_instances = training_instances[:options.training_sentence_size]

	# trim training set for size evaluation (token based)
	training_corpus_size = sum(training_vocab.values())
	if training_corpus_size > options.token_size:
		random.shuffle(training_instances)
		cumulative_tokens = 0
		cutoff_index = -1
		for i,inst in enumerate(training_instances):
			cumulative_tokens += len(inst.sentence)
			if cumulative_tokens >= options.token_size:
				training_instances = training_instances[:i+1]
				break
	
	# Add special tags to dicts
	for t2i in t2is.values():
		t2i[START_TAG] = len(t2i)
		t2i[END_TAG] = len(t2i)
	
	with open(options.output + ".train.pkl", "wb") as outfile:
		pickle.dump(training_instances, outfile)
	
	with open(options.output + ".dev.pkl", "wb") as outfile:
		pickle.dump(dev_instances, outfile)
	
	with open(options.output + ".test.pkl", "wb") as outfile:
		pickle.dump(test_instances, outfile)
	
	vocab_output = {"training_vocab": training_vocab, "w2i": w2i, "t2is": t2is, "c2i": c2i}
	with open(options.output + ".vocab.pkl", "wb") as outfile:
		pickle.dump(vocab_output, outfile)
	
	with open(options.vocab_file, "w", encoding="utf-8") as vocabfile:
		for word in w2i.keys():
			vocabfile.write(word + "\n")
