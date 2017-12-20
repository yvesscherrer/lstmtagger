'''
Main application script for training a tagger for parts-of-speech and morphosyntactic tags. Run with --help for command line arguments.
'''
from collections import Counter, defaultdict
from evaluate_morphotags import Evaluator

import os, sys, argparse, csv, datetime, collections
import random, pickle, progressbar, logging
import dynet as dy
import numpy as np
import utils

__author__ = "Yuval Pinter and Robert Guthrie, 2017 + Yves Scherrer"

Instance = collections.namedtuple("Instance", ["w_sentence", "c_sentence", "tags"])

UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
UNK_CHAR_TAG = "<?>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"

DEFAULT_WORD_EMBEDDING_SIZE = 64
DEFAULT_CHAR_EMBEDDING_SIZE = 20

def read_file(filename, w2i, t2is, c2i, vocab_counter, number_index=0, w_token_index=1, c_token_index=1, pos_index=3, morph_index=5, update_vocab=True):
	"""
	Read in a dataset and turn it into a list of instances.
	Modifies the w2i, t2is and c2i dicts, adding new words/attributes/tags/chars
	as it sees them.
	w_token_index: field in which the token representation used for the word embeddings is stored
	c_token_index: field in which the token representation used for the character embeddings is stored
	pos_index: field in which the main POS is stored
	morph_index: field in which the morphological tags are stored (-1 if no morphology)
	"""

	# populate mandatory t2i tables
	if POS_KEY not in t2is and update_vocab:
		t2is[POS_KEY] = {NONE_TAG: 0}

	# build dataset
	instances = []
	with open(filename, "r", encoding="utf-8") as f:

		# running sentence buffers (lines are tokens)
		w_sentence = []
		c_sentence = []
		tags = collections.defaultdict(list)
		idx = 0

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
				idx = 0

			else:

				# parse token information in line
				data = line.split("\t")
				if number_index < 0:
					idx += 1
				else:
					if '-' in data[number_index] or '.' in data[number_index]: # Some UD languages have contractions on a separate line, we don't want to include them also
						continue
					idx = int(data[number_index])
				w_word = data[w_token_index]
				c_word = data[c_token_index]
				if pos_index < 0:
					postag = NONE_TAG
				else:
					postag = data[pos_index] if pos_index != morph_index else data[pos_index].split("|")[0]
				morphotags = {} if morph_index < 0 else utils.split_tagstring(data[morph_index], uni_key=False, has_pos=(pos_index == morph_index))

				if update_vocab:
					# ensure counts and dictionary population
					vocab_counter[w_word] += 1
				
					if w_word not in w2i:
						w2i[w_word] = len(w2i)
					pt2i = t2is[POS_KEY]
					if postag not in pt2i:
						pt2i[postag] = len(pt2i)
					for c in c_word:
						if c not in c2i:
							c2i[c] = len(c2i)
					for key, val in morphotags.items():
						if key not in t2is:
							t2is[key] = {NONE_TAG:0}
						mt2i = t2is[key]
						if val not in mt2i:
							mt2i[val] = len(mt2i)

				# add data to sentence buffer
				w_sentence.append(w2i.get(w_word, w2i[UNK_TAG]))
				c_sentence.append([c2i[PADDING_CHAR]] + [c2i.get(c, c2i[UNK_CHAR_TAG]) for c in c_word] + [c2i[PADDING_CHAR]])
				tags[POS_KEY].append(t2is[POS_KEY].get(postag, t2is[POS_KEY][NONE_TAG]))
				for k,v in morphotags.items():
					if k not in t2is:
						# if there is an unknown morphological feature, we do not add it
						logging.info("Unknown feature key {} with value {} in file {} - skipping it".format(k, v, filename))
						continue
					
					mtags = tags[k]
					# pad backwards to latest seen
					missing_tags = idx - len(mtags) - 1
					mtags.extend([0] * missing_tags) # 0 guaranteed above to represent NONE_TAG
					mtags.append(t2is[k].get(v, NONE_TAG))
						
		
		# last sentence
		if len(w_sentence) > 0:
			# pad tag lists to sentence end
			slen = len(w_sentence)
			for seq in tags.values():
				if len(seq) < slen:
					seq.extend([0] * (slen - len(seq))) # 0 guaranteed below to represent NONE_TAG
			# add sentence to dataset
			instances.append(Instance(w_sentence, c_sentence, tags))
		
	return instances


class LSTMTagger:
	'''
	Joint POS/morphosyntactic attribute tagger based on LSTM.
	Embeddings are fed into Bi-LSTM layers, then hidden phases are fed into an MLP for each attribute type (including POS tags).
	Class "inspired" by Dynet's BiLSTM tagger tutorial script available at:
	https://github.com/clab/dynet_tutorial_examples/blob/master/tutorial_bilstm_tagger.py
	'''

	def __init__(self, tagset_sizes, num_lstm_layers, hidden_dim, word_embeddings, no_we_update, use_char_rnn, charset_size, char_embedding_dim, att_props=None, vocab_size=None, word_embedding_dim=None, populate_from_file=None, save_settings_to_file=None):
		'''
		:param tagset_sizes: dictionary of attribute_name:number_of_possible_tags
		:param num_lstm_layers: number of desired LSTM layers
		:param hidden_dim: size of hidden dimension (same for all LSTM layers, including character-level)
		:param word_embeddings: pre-trained list of embeddings, assumes order by word ID (optional)
		:param no_we_update: if toggled, don't update embeddings
		:param use_char_rnn: use "char->tag" option, i.e. concatenate character-level LSTM outputs to word representations (and train underlying LSTM). Only 1-layer is supported.
		:param charset_size: number of characters expected in dataset (needed for character embedding initialization)
		:param char_embedding_dim: desired character embedding dimension
		:param att_props: proportion of loss to assign each attribute for back-propagation weighting (optional)
		:param vocab_size: number of words in model (ignored if pre-trained embeddings are given)
		:param word_embedding_dim: desired word embedding dimension (ignored if pre-trained embeddings are given)
		:param populate_from_file: populate weights from saved file
		'''
		self.model = dy.ParameterCollection()	# ParameterCollection is the new name for Model
		self.tagset_sizes = tagset_sizes
		self.attributes = sorted(tagset_sizes.keys())
		self.we_update = not no_we_update
		if att_props is not None:
			self.att_props = defaultdict(float, {att:(1.0-p) for att,p in att_props.items()})
		else:
			self.att_props = None

		if word_embeddings is not None: # Use pretrained embeddings
			vocab_size = word_embeddings.shape[0]
			word_embedding_dim = word_embeddings.shape[1]
			logging.info("Use pretrained embeddings: setting vocabulary size to {} and dimensions to {}".format(vocab_size, word_embedding_dim))
		self.words_lookup = self.model.add_lookup_parameters((vocab_size, word_embedding_dim), name=b"we")

		if word_embeddings is not None:
			self.words_lookup.init_from_array(word_embeddings)

		# Char LSTM Parameters
		self.use_char_rnn = use_char_rnn
		if use_char_rnn:
			self.char_lookup = self.model.add_lookup_parameters((charset_size, char_embedding_dim), name=b"ce")
			self.char_bi_lstm = dy.BiRNNBuilder(1, char_embedding_dim, hidden_dim, self.model, dy.LSTMBuilder)

		# Word LSTM parameters
		if use_char_rnn:
			input_dim = word_embedding_dim + hidden_dim
		else:
			input_dim = word_embedding_dim
		self.word_bi_lstm = dy.BiRNNBuilder(num_lstm_layers, input_dim, hidden_dim, self.model, dy.LSTMBuilder)

		# Matrix that maps from Bi-LSTM output to num tags
		self.lstm_to_tags_params = {}
		self.lstm_to_tags_bias = {}
		self.mlp_out = {}
		self.mlp_out_bias = {}
		for att in self.attributes:	# need to be in consistent order for saving and loading
			set_size = tagset_sizes[att]
			self.lstm_to_tags_params[att] = self.model.add_parameters((set_size, hidden_dim), name=bytes(att+"H", 'utf-8'))
			self.lstm_to_tags_bias[att] = self.model.add_parameters(set_size, name=bytes(att+"Hb", 'utf-8'))
			self.mlp_out[att] = self.model.add_parameters((set_size, set_size), name=bytes(att+"O", 'utf-8'))
			self.mlp_out_bias[att] = self.model.add_parameters(set_size, name=bytes(att+"Ob", 'utf-8'))
		
		if save_settings_to_file:
			setting_dict = {
				"tagset_sizes": tagset_sizes,
				"num_lstm_layers": num_lstm_layers,
				"hidden_dim": hidden_dim,
				"word_embeddings": word_embeddings,
				"no_we_update": no_we_update,
				"use_char_rnn": use_char_rnn,
				"charset_size": charset_size,
				"char_embedding_dim": char_embedding_dim,
				"att_props": att_props,
				"vocab_size": vocab_size,
				"word_embedding_dim": word_embedding_dim
			}
			with open(save_settings_to_file, "wb") as outfile:
				pickle.dump(setting_dict, outfile)
		
		if populate_from_file:
			self.model.populate(populate_from_file)
	
		s = '''LSTM tagging model created with following parameters:

LSTM: {} layers, {} hidden dimensions
Word embeddings: {} dimensions
Concatenating character LSTM: {}
Char embeddings: {} dimensions
Vocabulary size: {}
Character set size: {}
Tagset sizes: {}
'''.format(num_lstm_layers, hidden_dim, word_embedding_dim, use_char_rnn, char_embedding_dim, vocab_size, charset_size, ", ".join(["{}:{}".format(x, tagset_sizes[x]) for x in sorted(tagset_sizes)]))
		logging.info(s)
		

	def word_rep(self, word, char_ids):
		'''
		:param word: index of word in lookup table
		'''
		wemb = dy.lookup(self.words_lookup, word, update=self.we_update)
		if char_ids is None:
			return wemb

		# add character representation
		char_embs = [self.char_lookup[cid] for cid in char_ids]
		char_exprs = self.char_bi_lstm.transduce(char_embs)
		return dy.concatenate([ wemb, char_exprs[-1] ])

	def build_tagging_graph(self, sentence, word_chars):
		dy.renew_cg()
		
		if not self.use_char_rnn:
			embeddings = [self.word_rep(w, None) for w in sentence]
		else:
			embeddings = [self.word_rep(w, chars) for w, chars in zip(sentence, word_chars)]

		lstm_out = self.word_bi_lstm.transduce(embeddings)

		H = {}
		Hb = {}
		O = {}
		Ob = {}
		scores = {}
		for att in self.attributes:
			H[att] = dy.parameter(self.lstm_to_tags_params[att])
			Hb[att] = dy.parameter(self.lstm_to_tags_bias[att])
			O[att] = dy.parameter(self.mlp_out[att])
			Ob[att] = dy.parameter(self.mlp_out_bias[att])
			scores[att] = []
			for rep in lstm_out:
				score_t = O[att] * dy.tanh(H[att] * rep + Hb[att]) + Ob[att]
				scores[att].append(score_t)

		return scores

	def loss(self, sentence, word_chars, tags_set):
		'''
		For use in training phase.
		Tag sentence (all attributes) and compute loss based on probability of expected tags.
		'''
		observations_set = self.build_tagging_graph(sentence, word_chars)
		errors = {}
		for att, tags in tags_set.items():
			err = []
			for obs, tag in zip(observations_set[att], tags):
				err_t = dy.pickneglogsoftmax(obs, tag)
				err.append(err_t)
			errors[att] = dy.esum(err)
		if self.att_props is not None:
			for att, err in errors.items():
				prop_vec = dy.inputVector([self.att_props[att]] * err.dim()[0])
				err = dy.cmult(err, prop_vec)
		return errors

	def tag_sentence(self, sentence, word_chars):
		'''
		For use in testing phase.
		Tag sentence and return tags for each attribute, without calculating loss.
		'''
		if not self.use_char_rnn:
			word_chars = None
		observations_set = self.build_tagging_graph(sentence, word_chars)
		tag_seqs = {}
		for att, observations in observations_set.items():
			observations = [ dy.softmax(obs) for obs in observations ]
			probs = [ obs.npvalue() for obs in observations ]
			tag_seq = []
			for prob in probs:
				tag_t = np.argmax(prob)
				tag_seq.append(tag_t)
			tag_seqs[att] = tag_seq
		return tag_seqs

	def set_dropout(self, p):
		self.word_bi_lstm.set_dropout(p)

	def disable_dropout(self):
		self.word_bi_lstm.disable_dropout()
		
	def save(self, file_name):
		'''
		Serialize model parameters for future loading and use.
		TODO change reading in scripts/test_model.py
		'''
		self.model.save(file_name)

	def old_save(self, file_name):
		'''
		Serialize model parameters for future loading and use.
		Old version (pre dynet 2.0) loaded using initializer in scripts/test_model.py
		'''
		members_to_save = []
		members_to_save.append(self.words_lookup)
		if (self.use_char_rnn):
			members_to_save.append(self.char_lookup)
			members_to_save.append(self.char_bi_lstm)
		members_to_save.append(self.word_bi_lstm)
		members_to_save.extend(utils.sortvals(self.lstm_to_tags_params))
		members_to_save.extend(utils.sortvals(self.lstm_to_tags_bias))
		members_to_save.extend(utils.sortvals(self.mlp_out))
		members_to_save.extend(utils.sortvals(self.mlp_out_bias))
		self.model.save(file_name, members_to_save)

		with open(file_name + "-atts", 'w') as attdict:
			attdict.write("\t".join(sorted(self.attributes)))


### END OF CLASSES ###

def get_att_prop(instances):
	logging.info("Calculating attribute proportions for proportional loss margin or proportional loss magnitude")
	total_tokens = 0
	att_counts = Counter()
	for instance in instances:
		total_tokens += len(instance.w_sentence)
		for att, tags in instance.tags.items():
			t2i = t2is[att]
			att_counts[att] += len([t for t in tags if t != t2i.get(NONE_TAG, -1)])
	return {att:(1.0 - (att_counts[att] / total_tokens)) for att in att_counts}


def evaluate(model, instances, outfilename, t2is, i2ts, i2w, i2c, training_vocab):
	model.disable_dropout()
	loss = 0.0
	correct = Counter()
	total = Counter()
	oov_total = Counter()
	bar = progressbar.ProgressBar()
	total_wrong = Counter()
	total_wrong_oov = Counter()
	f1_eval = Evaluator(m = 'att')
	display_eval = False
	if outfilename:
		writer = open(outfilename, 'w', encoding='utf-8')

	for instance in bar(instances):
		if len(instance.w_sentence) == 0: continue
		gold_tags = instance.tags
		for att in model.attributes:
			if att not in instance.tags:
				gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.w_sentence)
		losses = model.loss(instance.w_sentence, instance.c_sentence, gold_tags)
		total_loss = sum([l.scalar_value() for l in losses.values()])
		out_tags_set = model.tag_sentence(instance.w_sentence, instance.c_sentence)

		gold_strings = utils.morphotag_strings(i2ts, gold_tags)
		obs_strings = utils.morphotag_strings(i2ts, out_tags_set)
		for g, o in zip(gold_strings, obs_strings):
			f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
		for att, tags in gold_tags.items():
			# display the evaluation figures if we have ever seen a POS tag != NONE in the gold
			if (not display_eval) and (att == POS_KEY) and any([t != t2is[POS_KEY][NONE_TAG] for t in tags]):
				display_eval = True
			out_tags = out_tags_set[att]
			correct_sent = True

			oov_strings = []
			for word, gold, out in zip(instance.w_sentence, tags, out_tags):
				if gold == out:
					correct[att] += 1
				else:
					# Got the wrong tag
					total_wrong[att] += 1
					correct_sent = False
					if i2w[word] not in training_vocab:
						total_wrong_oov[att] += 1

				if i2w[word] not in training_vocab:
					oov_total[att] += 1
					oov_strings.append("OOV")
				else:
					oov_strings.append("")

			total[att] += len(tags)

		loss += (total_loss / len(instance.w_sentence))
		
		if writer:
			# regenerate output words from c_sentence, removing the padding characters
			out_sentence = [("".join([i2c[c] for c in w])).replace(PADDING_CHAR, "") for w in instance.c_sentence]
			writer.write("\n" + "\n".join(["\t".join(z) for z in zip(out_sentence, gold_strings, obs_strings, oov_strings)]) + "\n")
	
	if writer: writer.close()
	loss = loss / len(instances)

	# log results
	logging.info("Number of instances: {}".format(len(instances)))
	if display_eval:
		logging.info("POS Accuracy: {}".format(correct[POS_KEY] / total[POS_KEY]))
		logging.info("POS % OOV accuracy: {}".format((oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / oov_total[POS_KEY]))
		if total_wrong[POS_KEY] > 0:
			logging.info("POS % Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
		for attr in model.attributes:
			if attr != POS_KEY:
				logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
		logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), False))
	else:
		logging.info("Evaluation measures not available (no gold tags in file)")

	logging.info("Total tokens: {}, Total OOV: {}, % OOV: {}".format(total[POS_KEY], oov_total[POS_KEY], oov_total[POS_KEY] / total[POS_KEY]))
	logging.info("")
	return loss



if __name__ == "__main__":

	# ===-----------------------------------------------------------------------===
	# Argument parsing
	# ===-----------------------------------------------------------------------===
	parser = argparse.ArgumentParser()
	# input and output files
	parser.add_argument("--training-data", dest="training_data", help="File in which the training data is stored (either as CoNLLU text file or as saved pickle)")
	parser.add_argument("--training-data-save", dest="training_data_save", default=None, help="Pickle file in which the training data should be stored for future runs (if parameter is omitted, the training data is not saved as pickle)")
	parser.add_argument("--dev-data", dest="dev_data", help="File in which the dev data is stored (either as CoNLLU text file or as saved pickle)")
	parser.add_argument("--dev-data-save", dest="dev_data_save", default=None, help="Pickle file in which the dev data should be stored for future runs (if parameter is omitted, the dev data is not saved as pickle)")
	parser.add_argument("--dev-data-out", dest="dev_data_out", default=None, help="Text files in which to save the annotated dev files (epoch count is added in front of file extension)")
	parser.add_argument("--test-data", dest="test_data", help="File in which the test data is stored (either as CoNLLU text file or as saved pickle)")
	parser.add_argument("--test-data-save", dest="test_data_save", default=None, help="Pickle file in which the test data should be stored for future runs (if parameter is omitted, the test data is not saved as pickle)")
	parser.add_argument("--test-data-out", dest="test_data_out", default="testout.txt", help="Text file in which to save the output of the model")
	parser.add_argument("--vocab", dest="vocab", default=None, help="Pickle file from which an existing vocabulary is loaded")
	parser.add_argument("--vocab-save", dest="vocab_save", default=None, help="Pickle file in which the vocabulary is saved")
	parser.add_argument("--settings", dest="settings", default=None, help="Pickle file in which the model architecture is defined (if omitted, default settings are used)")
	parser.add_argument("--settings-save", dest="settings_save", default=None, help="Pickle file to which the model architecture is saved")
	parser.add_argument("--params", dest="params", default=None, help="Binary file (.bin) from which the model weights are loaded")
	parser.add_argument("--params-save", dest="params_save", default=None, help="Binary files (.bin) in which the model weights are saved (epoch count is added in front of file extension)")
	parser.add_argument("--word-embeddings", dest="word_embeddings", default=None, help="File from which to read in pretrained embeddings (if not supplied, will be random)")
	parser.add_argument("--log-dir", dest="log_dir", default="log", help="directory in which the log files are saved")
	
	# File format (default options are fine for UD-formatted files)
	# Call this script several times if not all datasets are formatted in the same way
	parser.add_argument("--number-index", dest="number_index", type=int, help="Field in which the word numbers are stored (default: 0)", default=0)
	parser.add_argument("--w-token-index", dest="w_token_index", type=int, help="Field in which the tokens used for the word embeddings are stored (default: 1)", default=1)
	parser.add_argument("--c-token-index", dest="c_token_index", type=int, help="Field in which the tokens used for the character embeddings are stored (default: 1)", default=1)
	parser.add_argument("--pos-index", dest="pos_index", type=int, help="Field in which the main POS is stored (default, UD tags: 3) (original non-UD tag: 4)", default=3)
	parser.add_argument("--morph-index", dest="morph_index", type=int, help="Field in which the morphology tags are stored (default: 5); use negative value if morphosyntactic tags should not be considered", default=5)
	
	# Options for easily reducing the size of the training data
	parser.add_argument("--training-sentence-size", default=sys.maxsize, dest="training_sentence_size", type=int, help="Instance count of training set (default: unlimited)")
	parser.add_argument("--training-token-size", default=sys.maxsize, dest="training_token_size", type=int, help="Token count of training set (default: unlimited)")
	
	# Model settings
	parser.add_argument("--num-epochs", default=20, dest="num_epochs", type=int, help="Number of full passes through training set (default: 20; disable training by setting --num-epochs to negative value)")
	parser.add_argument("--num-lstm-layers", default=2, dest="lstm_layers", type=int, help="Number of LSTM layers (default: 2)")
	parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers (default: 128)")
	parser.add_argument("--word-emb-dim", default=DEFAULT_WORD_EMBEDDING_SIZE, dest="word_emb_dim", type=int, help="Size of word embedding layer (ignored if pre-trained word embeddings are loaded)")
	parser.add_argument("--char-emb-dim", default=DEFAULT_CHAR_EMBEDDING_SIZE, dest="char_emb_dim", type=int, help="Size of character embedding layer (ignored if --use-char-rnn is not set)")
	parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate (default: 0.01)")
	parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="Amount of dropout to apply to LSTM part of graph (default: off)")
	parser.add_argument("--no-we-update", dest="no_we_update", action="store_true", help="Word Embeddings aren't updated")
	parser.add_argument("--loss-prop", dest="loss_prop", action="store_true", help="Proportional loss magnitudes")
	parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Use character RNN (default : off)")
	
	# other
	parser.add_argument("--dynet-mem", help="Ignore this external argument")
	parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
	options = parser.parse_args()
	
	# create log folder if required
	if not os.path.exists(options.log_dir):
		os.mkdir(options.log_dir)

	# Set up logging
	now = datetime.datetime.now()
	logging.basicConfig(filename=options.log_dir + "/run_{}{:02d}{:02d}_{:02d}{:02d}{:02d}.log".format(now.year, now.month, now.day, now.hour, now.minute, now.second), filemode="w", format="%(message)s", level=logging.INFO)
	
	
	if options.vocab:
		if not os.path.exists(options.vocab):
			logging.error("Vocabulary file does not exist at specified location: {}".format(options.options.vocab))
			sys.exit(1)
		else:
			logging.info("Load pickled vocabulary from {}".format(options.vocab))
			vocab_dataset = pickle.load(open(options.vocab, "rb"))
			training_vocab = vocab_dataset["training_vocab"]
			w2i = vocab_dataset["w2i"]
			t2is = vocab_dataset["t2is"]
			c2i = vocab_dataset["c2i"]
	else:
		training_vocab = None
				
	if options.training_data:
		if not os.path.exists(options.training_data):
			logging.error("Training data does not exist at specified location: {}".format(options.training_data))
			sys.exit(1)
			
		# read training data from pickle
		if options.training_data.endswith(".pkl"):
			logging.info("Load pickled training data from {}".format(options.training_data))
			training_instances = pickle.load(open(options.training_data, "rb"))
			if not training_vocab:
				logging.error("Cannot use pickled training data without corresponding vocabulary file (--vocab not set)")
				sys.exit(1)
		
		# read training data from text file
		else:
			if not training_vocab:
				training_vocab = collections.Counter()
				w2i = {UNK_TAG: 0} # mapping from word to index
				t2is = {} # mapping from attribute name to mapping from tag to index
				c2i = {UNK_CHAR_TAG: 0, PADDING_CHAR: 1} # mapping from character to index, for char-RNN concatenations
			# if we already have loaded the vocabulary, it should include these mappings
			
			logging.info("Read training data from {}".format(options.training_data))
			training_instances = read_file(options.training_data, w2i, t2is, c2i, training_vocab, options.number_index, options.w_token_index, options.c_token_index, options.pos_index, options.morph_index, update_vocab=True)
			
			# trim training set for size evaluation (sentence based)
			if len(training_instances) > options.training_sentence_size:
				logging.info("Reduce training sentence size to {}".format(options.training_sentence_size))
				random.shuffle(training_instances)
				training_instances = training_instances[:options.training_sentence_size]

			# trim training set for size evaluation (token based)
			training_corpus_size = sum(training_vocab.values())
			if training_corpus_size > options.training_token_size:
				logging.info("Reduce training token size to {}".format(options.training_token_size))
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
			
			logging.info("Training data loaded: {} instances, {} vocabulary items, {} characters, {} tag keys".format(len(training_instances), len(training_vocab), len(w2i), len(c2i), len(t2is)))		
			
		if options.training_data_save:
			logging.info("Save training data to {}".format(options.training_data_save))
			with open(options.training_data_save, "wb") as outfile:
				pickle.dump(training_instances, outfile)
			
		if options.vocab_save:
			logging.info("Save vocabulary to {}".format(options.vocab_save))
			vocab = {"training_vocab": training_vocab, "w2i": w2i, "t2is": t2is, "c2i": c2i}
			with open(options.vocab_save, "wb") as outfile:
				pickle.dump(vocab, outfile)
			
			#with open(options.vocab_save.replace(".pkl", ".txt"), "w", encoding="utf-8") as vocabfile:
			#	for word in w2i.keys():
			#		vocabfile.write(word + "\n")
	
	if options.dev_data:
		if not os.path.exists(options.dev_data):
			logging.error("Dev data does not exist at specified location: {}".format(options.dev_data))
			sys.exit(1)
		
		if not training_vocab:
			logging.error("Cannot use pickled dev data without corresponding vocabulary file (--vocab not set)")
			sys.exit(1)
		
		if options.dev_data.endswith(".pkl"):
			logging.info("Load pickled dev data from {}".format(options.dev_data))
			dev_instances = pickle.load(open(options.dev_data, "rb"))
		else:
			logging.info("Read dev data from {}".format(options.dev_data))
			dev_vocab = collections.Counter()
			dev_instances = read_file(options.dev_data, w2i, t2is, c2i, dev_vocab, options.number_index, options.w_token_index, options.c_token_index, options.pos_index, options.morph_index, update_vocab=False)
		
		logging.info("Dev data loaded: {} instances".format(len(dev_instances)))
		
		if options.dev_data_save:
			logging.info("Save training data to {}".format(options.dev_data_save))
			with open(options.dev_data_save, "wb") as outfile:
				pickle.dump(dev_instances, outfile)
		
		# debug samples small set for faster full loop
		if options.debug:
			dev_instances = dev_instances[0:int(len(dev_instances)/10)]
			logging.info("DEBUG MODE: reducing number of dev instances to {}".format(len(dev_instances)))
		
	else:
		dev_instances = None
		logging.info("No development data given (--dev-data not set)")
	
	if training_vocab:
		# Inverse vocabulary mapping
		i2w = { i: w for w, i in w2i.items() } # Inverse mapping
		i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }
		i2c = { i: c for c, i in c2i.items() }
		
	if options.num_epochs <= 0:
		model = None
		logging.info("Model training disabled (--num-epochs {})".format(options.num_epochs))
	
	# load pre-trained model
	elif options.settings or options.params:
		if not options.settings or not os.path.exists(options.settings):
			logging.error("Model setting file does not exist at specified location: {}".format(options.settings))
			sys.exit(1)
		if not options.params or not os.path.exists(options.params):
			logging.error("Model parameter file does not exist at specified location: {}".format(options.params))
			sys.exit(1)
		settings = pickle.load(open(options.settings, "rb"))
		model = LSTMTagger(tagset_sizes=settings["tagset_sizes"],
					num_lstm_layers=settings["num_lstm_layers"],
					hidden_dim=settings["hidden_dim"],
					word_embeddings=settings["word_embeddings"],
					no_we_update = settings["no_we_update"],
					use_char_rnn=settings["use_char_rnn"],
					charset_size=settings["charset_size"],
					char_embedding_dim=settings["char_embedding_dim"],
					att_props=settings["att_props"],
					vocab_size=settings["vocab_size"],
					word_embedding_dim=settings["word_embedding_dim"],
					populate_from_file=options.params)
		logging.info("Model loaded from files {} and {}".format(options.settings, options.params))
	
	# train new model
	else:
		logging.info("Create new model")
		tagset_sizes = { att: len(t2i) for att, t2i in t2is.items() }
		
		if options.word_embeddings:
			word_embeddings = utils.read_pretrained_embeddings(options.word_embeddings, w2i)
			logging.info("Using pretrained embeddings from file {}".format(options.word_embeddings))
		else:
			word_embeddings = None
		
		if options.loss_prop:
			att_props = get_att_prop(training_instances)
			logging.info("Using LSTM loss weights proportional to attribute frequency")
		else:
			att_props = None
			
		model = LSTMTagger(tagset_sizes=tagset_sizes,
						   num_lstm_layers=options.lstm_layers,
						   hidden_dim=options.hidden_dim,
						   word_embeddings=word_embeddings,
						   no_we_update = options.no_we_update,
						   use_char_rnn=options.use_char_rnn,
						   charset_size=len(c2i),
						   char_embedding_dim=DEFAULT_CHAR_EMBEDDING_SIZE,
						   att_props=att_props,
						   vocab_size=len(w2i),
						   word_embedding_dim=DEFAULT_WORD_EMBEDDING_SIZE,
						   save_settings_to_file=options.settings_save)
		
		########### start of training loop ###########
		
		trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9)
		logging.info("Starting training with algorithm: {}, epochs: {}, learning rate: {}, dropout: {}".format(type(trainer), options.num_epochs, options.learning_rate, options.dropout))
		train_dev_cost = csv.writer(open(options.log_dir + "/train_dev_loss.csv", 'w'))
		train_dev_cost.writerow(["Train_cost", "Dev_cost"])

		for epoch in range(int(options.num_epochs)):
			bar = progressbar.ProgressBar()

			# set up epoch
			random.shuffle(training_instances)
			train_loss = 0.0
			if options.dropout > 0:
				model.set_dropout(options.dropout)

			# debug samples small set for faster full loop
			if options.debug:
				train_instances = training_instances[0:int(len(training_instances)/20)]
				logging.info("DEBUG MODE: reducing number of train instances to {}".format(len(train_instances)))
			else:
				train_instances = training_instances

			# main training loop
			for instance in bar(train_instances):
				if len(instance.w_sentence) == 0: continue

				gold_tags = instance.tags
				for att in model.attributes:
					if att not in instance.tags:
						# 'pad' entire sentence with none tags
						gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.w_sentence)

				# calculate all losses for sentence
				loss_exprs = model.loss(instance.w_sentence, instance.c_sentence, gold_tags)
				loss_expr = dy.esum(list(loss_exprs.values()))
				loss = loss_expr.scalar_value()

				# bail if loss is NaN
				if np.isnan(loss):
					assert False, "NaN occured"

				train_loss += (loss / len(instance.w_sentence))

				# backward pass and parameter update
				loss_expr.backward()
				trainer.update()
			
			train_loss = train_loss / len(train_instances)

			# log epoch's train phase
			logging.info("")
			logging.info("Epoch {} complete".format(epoch + 1))
			logging.info("Number of training instances: {}".format(len(train_instances)))
			logging.info("Training Loss: {}".format(train_loss))
			logging.info("")
			# here used to be a learning rate update, no longer supported in dynet 2.0

			# evaluate dev data
			if dev_instances:
				if options.dev_data_out:
					if epoch+1 == options.num_epochs:
						devout_filename = options.dev_data_out
					else:
						devout_filename, devout_fileext = os.path.splitext(options.dev_data_out)
						devout_filename = "{}.epoch{:02d}{}".format(devout_filename, epoch+1, devout_fileext)
				else:
					devout_filename = None
				
				logging.info("Evaluate dev data")
				dev_loss = evaluate(model, dev_instances, devout_filename, t2is, i2ts, i2w, i2c, training_vocab)
				logging.info("Dev Loss: {}".format(dev_loss))

				if options.dev_data_out and epoch > 1 and epoch % 10 != 0: # leave outputs from epochs 1,10,20, etc.
					devout_filename, devout_fileext = os.path.splitext(options.dev_data_out)
					old_devout_filename = "{}.epoch{:02d}{}".format(devout_filename, epoch, devout_fileext)
					logging.info("Removing file from previous epoch: {}".format(old_devout_filename))
					os.remove(old_devout_filename)
				
				train_dev_cost.writerow([train_loss, dev_loss])
			else:
				train_dev_cost.writerow([train_loss, "NA"])

			# serialize model
			if options.params_save:
				if epoch+1 == options.num_epochs:
					param_filename = options.params_save
				else:
					param_filename, param_fileext = os.path.splitext(options.params_save)
					param_filename = "{}.epoch{:02d}{}".format(param_filename, epoch+1, param_fileext)
				logging.info("Saving model to {}".format(param_filename))
				model.save(param_filename)
				if epoch > 1 and epoch % 10 != 0: # leave models from epochs 1,10,20, etc.
					param_filename, param_fileext = os.path.splitext(options.params_save)
					old_param_filename = "{}.epoch{:02d}{}".format(param_filename, epoch, param_fileext)
					logging.info("Removing file from previous epoch: {}".format(old_param_filename))
					os.remove(old_param_filename)
					
			# epoch loop ends
		
		########### end of training loop ###########
	
	# by now we have loaded or trained a model
	
	if options.test_data:
		if not os.path.exists(options.test_data):
			logging.error("Test data does not exist at specified location: {}".format(options.test_data))
			sys.exit(1)
		
		if not training_vocab:
			logging.error("Cannot use pickled dev data without corresponding vocabulary file (--vocab not set)")
			sys.exit(1)
			
		if options.test_data.endswith(".pkl"):
			logging.info("Load pickled test data from {}".format(options.test_data))
			test_instances = pickle.load(open(options.test_data, "rb"))
		else:
			logging.info("Read test data from {}".format(options.test_data))
			test_vocab = collections.Counter()
			test_instances = read_file(options.test_data, w2i, t2is, c2i, test_vocab, options.number_index, options.w_token_index, options.c_token_index, options.pos_index, options.morph_index, update_vocab=False)
		
		logging.info("Test data loaded: {} instances".format(len(test_instances)))
			
		if options.test_data_save:
			logging.info("Save test data to {}".format(options.test_data_save))
			with open(options.test_data_save, "wb") as outfile:
				pickle.dump(test_instances, outfile)
		
		if options.debug:
			test_instances = test_instances[0:int(len(test_instances)/10)]
			logging.info("DEBUG MODE: reducing number of test instances to {}".format(len(test_instances)))
		
		if model:
			logging.info("Evaluate test data")
			evaluate(model, test_instances, options.test_data_out, t2is, i2ts, i2w, i2c, training_vocab)
	
	else:
		logging.info("No test data provided")
	
	logging.info("Finished")
	