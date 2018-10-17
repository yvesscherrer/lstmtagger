'''
Main application script for training a tagger for parts-of-speech and morphosyntactic tags. Run with --help for command line arguments.
'''
from collections import Counter, defaultdict
from evaluate_morphotags import Evaluator

import os, sys, argparse, csv, datetime, collections, itertools
import random, pickle, progressbar, logging
import dynet as dy
import numpy as np

__author__ = "Yves Scherrer, 2018, based on code by Yuval Pinter and Robert Guthrie, 2017"

Instance = collections.namedtuple("Instance", ["w_sentence", "c_sentence", "tags", "length"])

UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
UNK_CHAR_TAG = "<?>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"


def read_pretrained_embeddings(filename, w2i):
	embeddings = None
	updates = 0
	with open(filename, "r", encoding="utf-8") as f:
		for line in f:
			split = line.split()
			if len(split) > 2:
				word = split[0]
				try:
					vec = np.array(split[1:], dtype="float")
					if embeddings is None:	# create embeddings vector after seeing the first example, determining the nb of dimensions
						embedding_dim = len(vec)
						embeddings = np.random.uniform(-0.8, 0.8, (len(w2i), embedding_dim))
					if word in w2i and np.linalg.norm(vec) < 15.0:
						# There's a reason for this if condition.  Some tokens in ptb cause numerical problems because they 
						# are long strings of the same punctuation, e.g !!!!!!!!!!!!!!!!!!!!!!! which end up having huge norms,
						# since Morfessor will segment it as a ton of ! and then the sum of these morpheme vectors is huge.
						embeddings[w2i[word]] = vec
						updates += 1
				except ValueError:
					logging.info("Skip embedding starting with '{}'".format(" ".join(split[:4])))
	logging.info("{} out of {} vocabulary items associated with pretrained embeddings".format(updates, len(w2i)))
	return embeddings


def split_tagstring(s, uni_key=False, has_pos=False):
	'''
	Returns attribute-value mapping from UD-type CONLL field
	:param uni_key: if toggled, returns attribute-value pairs as joined strings (with the '=')
	:param has_pos: input line segment includes POS tag label
	'''
	if has_pos:
		s = s.split("\t")[1]
	ret = [] if uni_key else {}
	if "=" not in s: # incorrect format
		return ret
	for attval in s.split('|'):
		attval = attval.strip()
		if not uni_key:
			a,v = attval.split('=')
			ret[a] = v
		else:
			ret.append(attval)
	return ret


def map_tags(i2ts, tagdict, has_prob=False):
	tagstrdict = {}
	for attr, seq in tagdict.items():
		if has_prob:
			seq, prob = seq
			val = i2ts[attr][seq]
			if val != NONE_TAG:
				tagstrdict[attr] = (val, prob)
		else:
			val = i2ts[attr][seq]
			if val != NONE_TAG:
				tagstrdict[attr] = val
	return tagstrdict


def morphdict2str(morphdict, keepPos=False):
	l = sorted(["{}={}".format(m, morphdict[m]) for m in morphdict if m != POS_KEY])
	if keepPos:
		l = ["{}={}".format(POS_KEY, morphdict[POS_KEY])] + l
	if l == []:
		return "_"
	else:
		return "|".join(l)

def probdict2str(probdict, keepPos=False):
	l = sorted(["{}={:.3f}".format(m, probdict[m]) for m in probdict if m != POS_KEY])
	if keepPos:
		l = ["{}={:.3f}".format(POS_KEY, probdict[POS_KEY])] + l
	if l == []:
		return ""
	else:
		return "|".join(l)


def read_file(filename, w2i, t2is, c2i, c_vocab, number_index=0, w_token_index=1, c_token_index=1, pos_index=3, morph_index=5, update_vocab=True):
	"""
	Read in a dataset and turn it into a list of instances.
	Modifies the w2i, t2is, c2i and c_vocab dicts, adding new words/attributes/tags/chars
	as it sees them.
	w_token_index: field in which the token representation used for the word embeddings is stored (ignored if < 0)
	c_token_index: field in which the token representation used for the character embeddings is stored (ignored if < 0)
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
				slen = max(len(w_sentence), len(c_sentence))
				if slen > 0:
					for seq in tags.values():
						if len(seq) < slen:
							seq.extend([0] * (slen - len(seq))) # 0 guaranteed below to represent NONE_TAG
					# add sentence to dataset
					instances.append(Instance(w_sentence, c_sentence, tags, slen))
				
				w_sentence = []
				c_sentence = []
				tags = collections.defaultdict(list)
				idx = 0

			else:

				# parse token information in line
				data = [x.strip() for x in line.split("\t")]
				if number_index < 0:
					idx += 1
				else:
					if '-' in data[number_index] or '.' in data[number_index]: # Some UD languages have contractions on a separate line, we don't want to include them also
						continue
					idx = int(data[number_index])
				
				if w_token_index >= 0:
					w_word = data[w_token_index]
				else:
					w_word = None
				if c_token_index >= 0:
					c_word = data[c_token_index]
				else:
					c_word = None
				if pos_index >= 0:
					postag = data[pos_index] if pos_index != morph_index else data[pos_index].split("|")[0]
				else:
					postag = NONE_TAG
				if morph_index >= 0:
					morphotags = split_tagstring(data[morph_index], uni_key=False, has_pos=(pos_index == morph_index))
				else:
					morphotags = {}

				if update_vocab:
					if w_word is not None:
						if w_word not in w2i:
							w2i[w_word] = len(w2i)
					
					if c_word is not None:
						c_vocab.add(c_word)
						for c in c_word:
							if c not in c2i:
								c2i[c] = len(c2i)
					
					pt2i = t2is[POS_KEY]
					if postag not in pt2i:
						pt2i[postag] = len(pt2i)
										
					for key, val in morphotags.items():
						if key not in t2is:
							t2is[key] = {NONE_TAG:0}
						mt2i = t2is[key]
						if val not in mt2i:
							mt2i[val] = len(mt2i)

				# add data to sentence buffer
				if w_word is not None:
					w_sentence.append(w2i.get(w_word, w2i[UNK_TAG]))
				
				if c_word is not None:
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
					mtags.append(t2is[k].get(v, t2is[k][NONE_TAG]))
					
		# last sentence
		# pad tag lists to sentence end
		slen = max(len(w_sentence), len(c_sentence))
		if slen > 0:
			for seq in tags.values():
				if len(seq) < slen:
					seq.extend([0] * (slen - len(seq))) # 0 guaranteed below to represent NONE_TAG
			# add sentence to dataset
			instances.append(Instance(w_sentence, c_sentence, tags, slen))
		
	return instances


class LSTMTagger:
	'''
	Joint POS/morphosyntactic attribute tagger based on LSTM.
	Embeddings are fed into Bi-LSTM layers, then hidden phases are fed into an MLP for each attribute type (including POS tags).
	Class "inspired" by Dynet's BiLSTM tagger tutorial script available at:
	https://github.com/clab/dynet_tutorial_examples/blob/master/tutorial_bilstm_tagger.py
	'''

	def __init__(self, char_num_layers, char_set_size, char_embedding_dim, char_hidden_dim, word_pretrained_embeddings, word_set_size, word_embedding_dim, word_update_emb, tag_num_layers, tag_hidden_dim, tag_set_sizes, tag_att_props=None, populate_from_file=None, save_settings_to_file=None):
		'''
		:param char_num_layers: number of bi-LSTM layers for the character embedder (<1 if no char embedder is used)
		:param char_set_size: number of distinct characters expected in dataset
		:param char_embedding_dim: dimension of the character embeddings
		:param char_hidden_dim: dimension of the LSTM layers in the character embedder (same for all layers)
		:param word_pretrained_embeddings: pre-trained list of embeddings, assumes order by word ID (may be None)
		:param word_set_size: number of distinct words expected in dataset (is overridden by word_pretrained_embeddings)
		:param word_embedding_dim: dimension of the word embeddings (is overridden by word_pretrained_embeddings, <1 if no word embeddings are to be used)
		:param word_update_emb: if True, update the word embeddings
		:param tag_hidden_dim: dimension of the LSTM layers of the tagger (same for all layers)
		:param tag_set_sizes: dictionary of attribute_name:number_of_possible_tags
		:param tag_att_props: proportion of loss to assign each attribute for back-propagation weighting (optional)
		:param populate_from_file: populate weights from saved file (optional)
		:param save_settings_to_file: file name to which the settings are saved (optional)
		'''
		self.model = dy.ParameterCollection()	# ParameterCollection is the new name for Model
		self.tag_set_sizes = tag_set_sizes
		self.attributes = sorted(tag_set_sizes.keys())
		if tag_att_props is not None:
			self.att_props = defaultdict(float, {att:(1.0-p) for att,p in tag_att_props.items()})
		else:
			self.att_props = None
		
		self.use_char_lstm = char_num_layers > 0
		self.use_word_emb = (word_pretrained_embeddings is not None) or (word_embedding_dim > 0)

		tag_input_dim = 0
		if self.use_word_emb:
			if word_pretrained_embeddings is not None:
				word_set_size = word_pretrained_embeddings.shape[0]
				word_embedding_dim = word_pretrained_embeddings.shape[1]
				logging.info("Use pretrained embeddings: setting vocabulary size to {} and dimensions to {}".format(word_set_size, word_embedding_dim))
			self.word_lookup = self.model.add_lookup_parameters((word_set_size, word_embedding_dim))
			if word_pretrained_embeddings is not None:
				self.word_lookup.init_from_array(word_pretrained_embeddings)
			self.word_update_emb = word_update_emb
			tag_input_dim += word_embedding_dim

		if self.use_char_lstm:
			self.char_lookup = self.model.add_lookup_parameters((char_set_size, char_embedding_dim))
			self.char_bi_lstm = dy.BiRNNBuilder(char_num_layers, char_embedding_dim, char_hidden_dim, self.model, dy.LSTMBuilder)
			tag_input_dim += char_hidden_dim
		
		self.tag_bi_lstm = dy.BiRNNBuilder(tag_num_layers, tag_input_dim, tag_hidden_dim, self.model, dy.LSTMBuilder)

		# Matrix that maps from Bi-LSTM output to num tags
		self.lstm_to_tags_params = {}
		self.lstm_to_tags_bias = {}
		self.mlp_out = {}
		self.mlp_out_bias = {}
		for att in self.attributes:	# need to be in consistent order for saving and loading
			set_size = tag_set_sizes[att]
			self.lstm_to_tags_params[att] = self.model.add_parameters((set_size, tag_hidden_dim))
			self.lstm_to_tags_bias[att] = self.model.add_parameters(set_size)
			self.mlp_out[att] = self.model.add_parameters((set_size, set_size))
			self.mlp_out_bias[att] = self.model.add_parameters(set_size)
		
		if save_settings_to_file:
			setting_dict = {
				"char_num_layers": char_num_layers,
				"char_set_size": char_set_size,
				"char_embedding_dim": char_embedding_dim,
				"char_hidden_dim": char_hidden_dim,
				"word_pretrained_embeddings": word_pretrained_embeddings,
				"word_set_size": word_set_size,
				"word_embedding_dim": word_embedding_dim,
				"word_update_emb": word_update_emb,
				"tag_num_layers": tag_num_layers,
				"tag_hidden_dim": tag_hidden_dim,
				"tag_set_sizes": tag_set_sizes,
				"tag_att_props": tag_att_props
			}
			with open(save_settings_to_file, "wb") as outfile:
				pickle.dump(setting_dict, outfile)
		
		if populate_from_file:
			self.model.populate(populate_from_file)

		s = "LSTM tagging model created with following parameters:\n"
		
		if self.use_char_lstm:
			s += "- Character LSTM: {} characters, {} input embedding dimensions, {} layers, {} hidden dimensions per layer\n".format(char_set_size, char_embedding_dim, char_num_layers, char_hidden_dim)
		else:
			s += "- No character LSTM\n"
		
		if self.use_word_emb:
			s += "- Word embeddings: {} word types, {} embedding dimensions".format(word_set_size, word_embedding_dim)
			if word_pretrained_embeddings is not None:
				s += ", pretrained"
			if self.word_update_emb:
				s += ", update enabled"
			else:
				s += ", update disabled"
			s += "\n"
		else:
			s += "- No word embeddings\n"
		
		s += "- Tagging LSTM: {} layers, {} input dimensions, {} hidden dimensions per layer\n".format(tag_num_layers, tag_input_dim, tag_hidden_dim)
		s += "- Output tag set sizes: {}\n".format(str(tag_set_sizes))
		logging.info(s)
		

	def word_rep(self, w_token, c_token):
		if self.use_word_emb:
			wemb = dy.lookup(self.word_lookup, w_token, update=self.word_update_emb)
			if not self.use_char_lstm:
				return wemb
		if self.use_char_lstm:
			char_embs = [self.char_lookup[c] for c in c_token]
			char_exprs = self.char_bi_lstm.transduce(char_embs)
			if not self.use_word_emb:
				return char_exprs[-1]
		return dy.concatenate([wemb, char_exprs[-1]])


	def build_tagging_graph(self, w_sentence, c_sentence):
		dy.renew_cg()
		
		if self.use_word_emb:
			if self.use_char_lstm:
				embeddings = [self.word_rep(w, c) for w, c in zip(w_sentence, c_sentence)]
			else:
				embeddings = [self.word_rep(w, None) for w in w_sentence]
		elif self.use_char_lstm:
			embeddings = [self.word_rep(None, c) for c in c_sentence]
		lstm_out = self.tag_bi_lstm.transduce(embeddings)

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

	def loss(self, w_sentence, c_sentence, tags_set):
		'''
		For use in training phase.
		Tag sentence (all attributes) and compute loss based on probability of expected tags.
		'''
		observations_set = self.build_tagging_graph(w_sentence, c_sentence)
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

	def tag_sentence(self, w_sentence, c_sentence):
		'''
		For use in testing phase.
		Tag sentence and return tags for each attribute, without calculating loss.
		'''
		observations_set = self.build_tagging_graph(w_sentence, c_sentence)
		tag_seqs = {}
		for att, observations in observations_set.items():
			observations = [ dy.softmax(obs) for obs in observations ]
			probs = [ obs.npvalue() for obs in observations ]
			tag_seq = []
			for prob in probs:
				tag_t = np.argmax(prob)
				tag_seq.append((tag_t, prob[tag_t]))
			tag_seqs[att] = tag_seq
		return tag_seqs

	def set_dropout(self, p):
		if hasattr(self, 'char_bi_lstm'):
			self.char_bi_lstm.set_dropout(p)
		self.tag_bi_lstm.set_dropout(p)

	def disable_dropout(self):
		if hasattr(self, 'char_bi_lstm'):
			self.char_bi_lstm.disable_dropout()
		self.tag_bi_lstm.disable_dropout()
		
	def save(self, file_name):
		self.model.save(file_name)


### END OF CLASSES ###

def get_att_prop(instances):
	logging.info("Calculating attribute proportions for proportional loss margin or proportional loss magnitude")
	total_tokens = 0
	att_counts = Counter()
	for instance in instances:
		total_tokens += instance.length
		for att, tags in instance.tags.items():
			t2i = t2is[att]
			att_counts[att] += len([t for t in tags if t != t2i.get(NONE_TAG, -1)])
	return {att:(1.0 - (att_counts[att] / total_tokens)) for att in att_counts}


def evaluate(model, instances, outfilename, t2is, i2ts, i2w, i2c, c_vocab, no_eval_feats=[], add_probs=False):
	bar = progressbar.ProgressBar()
	model.disable_dropout()
	loss = 0.0
	eval = Evaluator(m='att')
	oov_eval = Evaluator(m='att')
	display_eval = False
	
	if outfilename:
		writer = open(outfilename, 'w', encoding='utf-8')
	else:
		writer = None
	
	for instance in bar(instances):
		# Instance(w_sentence, c_sentence, tags, length)
		if instance.length == 0:
			continue
		predicted_tags = model.tag_sentence(instance.w_sentence, instance.c_sentence)
		losses = model.loss(instance.w_sentence, instance.c_sentence, instance.tags)
		total_loss = sum([l.scalar_value() for l in losses.values()]) / instance.length
		loss += total_loss
		
		for i, (w_word, c_word) in enumerate(itertools.zip_longest(instance.w_sentence, instance.c_sentence)):
			is_oov = True

			if w_word and w_word in i2w and i2w[w_word] != UNK_TAG:
				w_word_str = i2w[w_word]
				is_oov = False
			else:
				w_word_str = UNK_TAG
			
			if c_word:
				# regenerate output words from c_sentence, removing the padding characters
				c_word_str = ("".join([i2c[c] for c in c_word])).replace(PADDING_CHAR, "")
				if c_word_str in c_vocab:
					is_oov = False
			else:
				c_word_str = UNK_TAG

			gold_tags = map_tags(i2ts, {x: instance.tags[x][i] for x in instance.tags})
			pred_tags_probs = map_tags(i2ts, {x: predicted_tags[x][i] for x in predicted_tags}, has_prob=True)
			pred_tags = {x: pred_tags_probs[x][0] for x in pred_tags_probs}
			if add_probs:
				pred_probs = {x: pred_tags_probs[x][1] for x in pred_tags_probs}
			eval_gold_tags = {x: gold_tags[x] for x in gold_tags if x not in no_eval_feats}
			eval_pred_tags = {x: pred_tags[x] for x in pred_tags if x not in no_eval_feats}
			eval.add_instance(eval_gold_tags, eval_pred_tags)
			if is_oov:
				oov_eval.add_instance(eval_gold_tags, eval_pred_tags)
			# display the evaluation figures if we have ever seen a POS tag != NONE in the gold
			if (not display_eval) and (POS_KEY in gold_tags):
				display_eval = True
			
			if writer:
				line = []		# line = [word, pos, morph, oov]
				if w_word_str != UNK_TAG:
					line.append(w_word_str)
				else:
					line.append(c_word_str)
				line.append(pred_tags.get(POS_KEY, "_"))
				line.append(morphdict2str(pred_tags))
				if is_oov:
					line.append("OOV")
				else:
					line.append("")
				if add_probs:
					line.append(probdict2str(pred_probs, keepPos=True))
				writer.write("\t".join(line) + "\n")
		
		if writer:
			writer.write("\n")
	
	if writer:
		writer.close()
	loss = loss / len(instances)
				
	# log results
	logging.info("")
	logging.info("{} instances".format(len(instances)))
	logging.info("{} tokens".format(eval.instance_count))
	logging.info("{} OOV tokens ({:.2f}%)".format(oov_eval.instance_count, 100 * oov_eval.instance_count / eval.instance_count))
	
	if display_eval:
		logging.info("")
		logging.info("Feature\tOverall F1\tOOV F1")
		logging.info("POS\t{:.2f}%\t{:.2f}%".format(100*eval.mic_f1(att=POS_KEY), 100*oov_eval.mic_f1(att=POS_KEY)))
		for attr in model.attributes:
			if (attr == POS_KEY) or (attr in no_eval_feats):
				continue
			logging.info("{}\t{:.2f}%\t{:.2f}%".format(attr, 100*eval.mic_f1(att=attr), 100*oov_eval.mic_f1(att=attr)))
		logging.info("Total Micro F1 (including POS)\t{:.2f}%\t{:.2f}%".format(100*eval.mic_f1(), 100*oov_eval.mic_f1()))
		logging.info("Total Micro F1 (excluding POS)\t{:.2f}%\t{:.2f}%".format(100*eval.mic_f1(excl=POS_KEY), 100*oov_eval.mic_f1(excl=POS_KEY)))
		logging.info("Total Macro F1 (including POS)\t{:.2f}%\t{:.2f}%".format(100*eval.mac_f1(), 100*oov_eval.mac_f1()))
		logging.info("Total Macro F1 (excluding POS)\t{:.2f}%\t{:.2f}%".format(100*eval.mac_f1(excl=POS_KEY), 100*oov_eval.mac_f1(excl=POS_KEY)))
	else:
		logging.info("Evaluation measures not available (no gold tags in file)")
	
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
	parser.add_argument("--keep-every", dest="keep_every", type=int, default=10, help="Keep model files and dev output every n epochs (default: 10, 0 only saves last)")
	parser.add_argument("--log-dir", dest="log_dir", default="log", help="directory in which the log files are saved")
	
	# File format (default options are fine for UD-formatted files)
	# Call this script several times if not all datasets are formatted in the same way
	parser.add_argument("--number-index", dest="number_index", type=int, help="Field in which the word numbers are stored (default: 0)", default=0)
	parser.add_argument("--w-token-index", dest="w_token_index", type=int, help="Field in which the tokens used for the word embeddings are stored (default: 1)", default=1)
	parser.add_argument("--c-token-index", dest="c_token_index", type=int, help="Field in which the tokens used for the character embeddings are stored (default: 1)", default=1)
	parser.add_argument("--pos-index", dest="pos_index", type=int, help="Field in which the main POS is stored (default, UD tags: 3) (original non-UD tag: 4)", default=3)
	parser.add_argument("--morph-index", dest="morph_index", type=int, help="Field in which the morphology tags are stored (default: 5); use negative value if morphosyntactic tags should not be considered", default=5)
	parser.add_argument("--no-eval-feats", dest="no_eval_feats", type=str, help="(Comma-separated) list of morphological features that should be ignored during evaluation; typically used for additional tasks in multitask settings", default="")
	parser.add_argument("--add-probs", dest="add_probs", action="store_true", help="Write prediction probabilities to output files")
	
	# Options for easily reducing the size of the training data
	parser.add_argument("--training-sentence-size", default=sys.maxsize, dest="training_sentence_size", type=int, help="Instance count of training set (default: unlimited)")
	# parser.add_argument("--training-token-size", default=sys.maxsize, dest="training_token_size", type=int, help="Token count of training set (default: unlimited)")

	# Model settings
	parser.add_argument("--num-epochs", default=40, dest="num_epochs", type=int, help="Number of full passes through training set (default: 40; disable training by setting --num-epochs to negative value)")
	parser.add_argument("--char-num-layers", default=2, dest="char_num_layers", type=int, help="Number of character LSTM layers (default: 2, use 0 to disable character LSTM)")
	parser.add_argument("--char-emb-dim", default=128, dest="char_embedding_dim", type=int, help="Size of character embedding layer (default: 128)")
	parser.add_argument("--char-hidden-dim", default=256, dest="char_hidden_dim", type=int, help="Size of character LSTM hidden layers (default: 256)")
	parser.add_argument("--word-emb-dim", default=256, dest="word_embedding_dim", type=int, help="Size of word embedding layer (ignored if pre-trained word embeddings are loaded, use 0 to disable word embeddings)")
	parser.add_argument("--pretrained-embeddings", dest="word_pretrained_embeddings", default=None, help="File from which to read in pretrained embeddings (if not supplied, will be random)")
	parser.add_argument("--fix-embeddings", dest="fix_embeddings", action="store_true", help="Do not update word embeddings during training (default: off, only makes sense with pretrained embeddings)")
	parser.add_argument("--tag-num-layers", default=2, dest="tag_num_layers", type=int, help="Number of tagger LSTM layers (default: 2)")
	parser.add_argument("--tag-hidden-dim", default=256, dest="tag_hidden_dim", type=int, help="Size of tagger LSTM hidden layers (default: 256)")
	parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate (default: 0.01)")
	parser.add_argument("--decay", default=0.1, dest="decay", type=float, help="Learning rate decay (default: 0.1, 0 to turn off)")
	parser.add_argument("--dropout", default=0.02, dest="dropout", type=float, help="Amount of dropout to apply to LSTM parts of graph (default: 0.02, -1 to turn off)")
	parser.add_argument("--loss-prop", dest="loss_prop", action="store_true", help="Proportional loss magnitudes")
	
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
	
	# make sure that the given parameters are consistent
	if options.char_num_layers < 1 and options.c_token_index >= 0:
		logging.info("Setting c_token_index from {} to -1 because char_num_layers is set to {}".format(options.c_token_index, options.char_num_layers))
		options.c_token_index = -1
	if options.c_token_index < 0 and options.char_num_layers > 0:
		logging.info("Setting char_num_layers from {} to -1 because c_token_index is set to {}".format(options.char_num_layers, options.c_token_index))
		options.char_num_layers = -1
	if options.word_embedding_dim < 1 and options.word_pretrained_embeddings is None and options.w_token_index >= 0:
		logging.info("Setting w_token_index from {} to -1 because word_embedding_dim is set to {} and word_pretrained_embeddings is not given".format(options.w_token_index, options.word_embedding_dim))
		options.w_token_index = -1
	if options.w_token_index < 0 and options.word_pretrained_embeddings is not None:
		logging.info("Setting word_pretrained_embeddings from {} to None because w_token_index is set to {}".format(options.word_pretrained_embeddings, options.w_token_index))
		options.word_pretrained_embeddings = None
	if options.w_token_index < 0 and options.word_embedding_dim > 0:
		logging.info("Setting word_embedding_dim from {} to -1 because w_token_index is set to {}".format(options.word_embedding_dim, options.w_token_index))
		options.word_embedding_dim = -1
	options.no_eval_feats = [x for x in options.no_eval_feats.strip().split(",") if x != ""]
	if options.no_eval_feats != []:
		logging.info("Disregard the following features in evaluation: {}".format(",".join(options.no_eval_feats)))
	
	if options.vocab:
		if not os.path.exists(options.vocab):
			logging.error("Vocabulary file does not exist at specified location: {}".format(options.vocab))
			sys.exit(1)
		else:
			logging.info("Load pickled vocabulary from {}".format(options.vocab))
			vocab_dataset = pickle.load(open(options.vocab, "rb"))
			w2i = vocab_dataset["w2i"]
			t2is = vocab_dataset["t2is"]
			c2i = vocab_dataset["c2i"]
			# if is for backwards compatibility reason
			if "c_vocab" in vocab_dataset:
				c_vocab = vocab_dataset["c_vocab"]
			elif "training_vocab" in vocab_dataset:
				logging.info("Converting training_vocab to c_vocab - OOV counts may be inaccurate")
				c_vocab = set(vocab_dataset["training_vocab"].keys())
			else:
				logging.info("c_vocab not found (old model) - OOV counts may be inaccurate")
				c_vocab = set()
	else:
		w2i = None
		t2is = None
		c2i = None
		c_vocab = None
				
	if options.training_data:
		if not os.path.exists(options.training_data):
			logging.error("Training data does not exist at specified location: {}".format(options.training_data))
			sys.exit(1)
			
		# read training data from pickle
		if options.training_data.endswith(".pkl"):
			logging.info("Load pickled training data from {}".format(options.training_data))
			training_instances = pickle.load(open(options.training_data, "rb"))
			if not training_vocab:
				if options.word_embedding_dim > 0 or options.word_pretrained_embeddings:
					logging.error("Cannot use pickled training data without corresponding vocabulary file (--vocab not set)")
					sys.exit(1)
				else:
					logging.info("Vocabulary file not found (--vocab not set), OOV rates will be inaccurate")
		
		# read training data from text file
		else:
			if not w2i:
				w2i = {UNK_TAG: 0} # mapping from word to index
				t2is = {} # mapping from attribute name to mapping from tag to index
				c2i = {UNK_CHAR_TAG: 0, PADDING_CHAR: 1} # mapping from character to index, for char-RNN concatenations
				c_vocab = set()	# set of words as seen in the character-level representation
			# if we already have loaded the vocabulary, it should include these mappings
			
			logging.info("Read training data from {}".format(options.training_data))
			training_instances = read_file(options.training_data, w2i, t2is, c2i, c_vocab, options.number_index, options.w_token_index, options.c_token_index, options.pos_index, options.morph_index, update_vocab=True)
			
			# trim training set for size evaluation (sentence based)
			if len(training_instances) > options.training_sentence_size:
				logging.info("Reduce training sentence size to {}".format(options.training_sentence_size))
				random.shuffle(training_instances)
				training_instances = training_instances[:options.training_sentence_size]

			# trim training set for size evaluation (token based)
			# training_corpus_size = sum(training_vocab.values())
			# if training_corpus_size > options.training_token_size:
				# logging.info("Reduce training token size to {}".format(options.training_token_size))
				# random.shuffle(training_instances)
				# cumulative_tokens = 0
				# cutoff_index = -1
				# for i,inst in enumerate(training_instances):
					# cumulative_tokens += len(inst.sentence)
					# if cumulative_tokens >= options.token_size:
						# training_instances = training_instances[:i+1]
						# break
			
			logging.info("Training data loaded: {} instances, {} word-embedding vocabulary items, {} character-embedding vocabulary items, {} characters, {} tag keys".format(len(training_instances), len(w2i), len(c_vocab), len(c2i), len(t2is)))		
			
		if options.training_data_save:
			logging.info("Save training data to {}".format(options.training_data_save))
			with open(options.training_data_save, "wb") as outfile:
				pickle.dump(training_instances, outfile)
			
		if options.vocab_save:
			logging.info("Save vocabulary to {}".format(options.vocab_save))
			vocab = {"c_vocab": c_vocab, "w2i": w2i, "t2is": t2is, "c2i": c2i}
			with open(options.vocab_save, "wb") as outfile:
				pickle.dump(vocab, outfile)
	
	if options.dev_data:
		if not os.path.exists(options.dev_data):
			logging.error("Dev data does not exist at specified location: {}".format(options.dev_data))
			sys.exit(1)
		
		if not w2i:
			if options.word_embedding_dim > 0 or options.word_pretrained_embeddings:
				logging.error("Cannot use pickled training data without corresponding vocabulary file (--vocab not set)")
				sys.exit(1)
			else:
				logging.info("Vocabulary file not found (--vocab not set), OOV rates will be inaccurate")
		
		if options.dev_data.endswith(".pkl"):
			logging.info("Load pickled dev data from {}".format(options.dev_data))
			dev_instances = pickle.load(open(options.dev_data, "rb"))
		else:
			logging.info("Read dev data from {}".format(options.dev_data))
			dev_instances = read_file(options.dev_data, w2i, t2is, c2i, c_vocab, options.number_index, options.w_token_index, options.c_token_index, options.pos_index, options.morph_index, update_vocab=False)
		
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
	
	if w2i:
		# Inverse vocabulary mapping
		i2w = { i: w for w, i in w2i.items() }
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
		model = LSTMTagger(char_num_layers = settings["char_num_layers"],
			char_set_size = settings["char_set_size"],
			char_embedding_dim = settings["char_embedding_dim"],
			char_hidden_dim = settings["char_hidden_dim"],
			word_pretrained_embeddings = settings["word_pretrained_embeddings"],
			word_set_size = settings["word_set_size"],
			word_embedding_dim = settings["word_embedding_dim"],
			word_update_emb = settings["word_update_emb"],
			tag_num_layers = settings["tag_num_layers"],
			tag_hidden_dim = settings["tag_hidden_dim"],
			tag_set_sizes = settings["tag_set_sizes"],
			tag_att_props = settings["tag_att_props"], 
			populate_from_file = options.params)		
		logging.info("Model loaded from files {} and {}".format(options.settings, options.params))
	
	# train new model
	else:
		logging.info("Create new model")
		tag_set_sizes = { att: len(t2i) for att, t2i in t2is.items() }
		
		if options.word_pretrained_embeddings:
			word_embeddings = read_pretrained_embeddings(options.word_pretrained_embeddings, w2i)
			logging.info("Using pretrained embeddings from file {}".format(options.word_pretrained_embeddings))
		else:
			word_embeddings = None
		
		if options.loss_prop:
			att_props = get_att_prop(training_instances)
			logging.info("Using LSTM loss weights proportional to attribute frequency")
		else:
			att_props = None
			
		model = LSTMTagger(char_num_layers = options.char_num_layers,
							char_set_size = len(c2i),
							char_embedding_dim = options.char_embedding_dim,
							char_hidden_dim = options.char_hidden_dim,
							word_pretrained_embeddings = word_embeddings,
							word_set_size = len(w2i),
							word_embedding_dim = options.word_embedding_dim,
							word_update_emb = not options.fix_embeddings,
							tag_num_layers = options.tag_num_layers,
							tag_hidden_dim = options.tag_hidden_dim,
							tag_set_sizes = tag_set_sizes,
							tag_att_props=att_props,
							save_settings_to_file=options.settings_save)
		
		########### start of training loop ###########
		
		trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9)
		logging.info("Starting training with algorithm: {}, epochs: {}, initial learning rate: {}, decay: {}, dropout: {}".format(type(trainer).__name__, options.num_epochs, options.learning_rate, options.decay, options.dropout))
		train_dev_cost = csv.writer(open(options.log_dir + "/train_dev_loss.csv", 'w'))
		train_dev_cost.writerow(["Train_cost", "Dev_cost"])

		for epoch in range(1, options.num_epochs + 1):		# epoch counts starts at 1 and includes options.num_epochs
			logging.info("Starting training epoch {}".format(epoch))
			bar = progressbar.ProgressBar()

			# set up epoch
			random.shuffle(training_instances)
			train_loss = 0.0
			if options.dropout > 0:
				model.set_dropout(options.dropout)
			is_last_epoch = (epoch == options.num_epochs)

			# debug samples small set for faster full loop
			if options.debug:
				train_instances = training_instances[0:int(len(training_instances)/20)]
				logging.info("DEBUG MODE: reducing number of train instances to {}".format(len(train_instances)))
			else:
				train_instances = training_instances

			# main training loop
			for instance in bar(train_instances):
				if instance.length == 0: continue

				gold_tags = instance.tags
				for att in model.attributes:
					if att not in instance.tags:
						# 'pad' entire sentence with none tags
						gold_tags[att] = [t2is[att][NONE_TAG]] * instance.length

				# calculate all losses for sentence
				loss_exprs = model.loss(instance.w_sentence, instance.c_sentence, gold_tags)
				loss_expr = dy.esum(list(loss_exprs.values()))
				loss = loss_expr.scalar_value()

				# bail if loss is NaN
				if np.isnan(loss):
					assert False, "NaN occured"

				train_loss += (loss / instance.length)

				# backward pass and parameter update
				loss_expr.backward()
				trainer.update()
			
			train_loss = train_loss / len(train_instances)

			# log epoch's train phase
			logging.info("Number of training instances: {}".format(len(train_instances)))
			logging.info("Training Loss: {}".format(train_loss))
			logging.info("")

			# evaluate dev data
			if dev_instances:
				if options.dev_data_out:
					if is_last_epoch:
						devout_filename = options.dev_data_out
					else:
						devout_filename, devout_fileext = os.path.splitext(options.dev_data_out)
						devout_filename = "{}.epoch{:02d}{}".format(devout_filename, epoch, devout_fileext)
				else:
					devout_filename = None
				
				logging.info("Evaluate dev data")
				dev_loss = evaluate(model, dev_instances, devout_filename, t2is, i2ts, i2w, i2c, c_vocab, options.no_eval_feats, options.add_probs)
				logging.info("Dev Loss: {}".format(dev_loss))
				
				# remove output of previous epoch if (a) there is output and we're not in the first epoch, and (b1) we don't want to save anything, or (b2) it is not the second epoch and it is not one of the epochs we want to save
				if options.dev_data_out and epoch > 1:
					if (options.keep_every <= 0) or ((epoch > 2) and ((epoch-1) % options.keep_every != 0)):
						devout_filename, devout_fileext = os.path.splitext(options.dev_data_out)
						old_devout_filename = "{}.epoch{:02d}{}".format(devout_filename, epoch-1, devout_fileext)
						logging.info("Removing file from previous epoch: {}".format(old_devout_filename))
						os.remove(old_devout_filename)
				
				train_dev_cost.writerow([train_loss, dev_loss])
			else:
				train_dev_cost.writerow([train_loss, "NA"])

			# serialize model
			if options.params_save:
				if is_last_epoch:
					param_filename = options.params_save
				else:
					param_filename, param_fileext = os.path.splitext(options.params_save)
					param_filename = "{}.epoch{:02d}{}".format(param_filename, epoch, param_fileext)
				logging.info("Saving model to {}".format(param_filename))
				model.save(param_filename)
				if options.params_save and epoch > 1:
					if (options.keep_every <= 0) or ((epoch > 2) and ((epoch-1) % options.keep_every != 0)):
						param_filename, param_fileext = os.path.splitext(options.params_save)
						old_param_filename = "{}.epoch{:02d}{}".format(param_filename, epoch-1, param_fileext)
						logging.info("Removing file from previous epoch: {}".format(old_param_filename))
						os.remove(old_param_filename)
			
			# learning rate update
			temp = trainer.learning_rate
			trainer.learning_rate *= 1.0-options.decay		# this was /= but this makes the learning rate increase, which shouldn't be?
			logging.info("Change learning rate from {} to {}".format(temp, trainer.learning_rate))
			logging.info("Epoch {} completed".format(epoch))
			logging.info("")
			# epoch loop ends
		
		########### end of training loop ###########
	
	# by now we have loaded or trained a model
	
	if options.test_data:
		if not os.path.exists(options.test_data):
			logging.error("Test data does not exist at specified location: {}".format(options.test_data))
			sys.exit(1)
		
		if not w2i:
			logging.error("Cannot use pickled dev data without corresponding vocabulary file (--vocab not set)")
			sys.exit(1)
			
		if options.test_data.endswith(".pkl"):
			logging.info("Load pickled test data from {}".format(options.test_data))
			test_instances = pickle.load(open(options.test_data, "rb"))
		else:
			logging.info("Read test data from {}".format(options.test_data))
			test_instances = read_file(options.test_data, w2i, t2is, c2i, c_vocab, options.number_index, options.w_token_index, options.c_token_index, options.pos_index, options.morph_index, update_vocab=False)
		
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
			evaluate(model, test_instances, options.test_data_out, t2is, i2ts, i2w, i2c, c_vocab, options.no_eval_feats, options.add_probs)
	
	else:
		logging.info("No test data provided")
	
	logging.info("Finished")
	