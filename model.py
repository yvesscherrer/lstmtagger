'''
Main application script for training a tagger for parts-of-speech and morphosyntactic tags. Run with --help for command line arguments.
'''
from collections import Counter, defaultdict
from evaluate_morphotags import Evaluator
from make_dataset import Instance

#import collections
import argparse
import random
import pickle
import logging
import progressbar
import os
import csv
import dynet as dy
import numpy as np

import utils

__author__ = "Yuval Pinter and Robert Guthrie, 2017 + Yves Scherrer"

#Instance = collections.namedtuple("Instance", ["w_sentence", "c_sentence", "tags"])

NONE_TAG = "<NONE>"
PADDING_CHAR = "<*>"
POS_KEY = "POS"

DEFAULT_WORD_EMBEDDING_SIZE = 64
DEFAULT_CHAR_EMBEDDING_SIZE = 20

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
		print(vocab_size, word_embedding_dim)
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
			with open(options.model_dir + "/" + save_settings_to_file, "wb") as outfile:
				pickle.dump(setting_dict, outfile)
		
		if populate_from_file:
			self.model.populate(populate_from_file)

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
		Tag sentence and return tags for each attribute, without caluclating loss.
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
	
	with open(outfilename, 'w', encoding='utf-8') as writer:
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

			# regenerate output words from c_sentence, removing the padding characters
			out_sentence = [("".join([i2c[c] for c in w])).replace(PADDING_CHAR, "") for w in instance.c_sentence]
			writer.write("\n" + "\n".join(["\t".join(z) for z in zip(out_sentence, gold_strings, obs_strings, oov_strings)]) + "\n")

	loss = loss / len(instances)

	# log results
	logging.info("Number of instances: {}".format(len(instances)))
	logging.info("POS Accuracy: {}".format(correct[POS_KEY] / total[POS_KEY]))
	logging.info("POS % OOV accuracy: {}".format((oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / oov_total[POS_KEY]))
	if total_wrong[POS_KEY] > 0:
		logging.info("POS % Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
	for attr in model.attributes:
		if attr != POS_KEY:
			logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
	logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), False))

	logging.info("Total tokens: {}, Total OOV: {}, % OOV: {}".format(total[POS_KEY], oov_total[POS_KEY], oov_total[POS_KEY] / total[POS_KEY]))
	return loss


if __name__ == "__main__":

	# ===-----------------------------------------------------------------------===
	# Argument parsing
	# ===-----------------------------------------------------------------------===
	parser = argparse.ArgumentParser()
	# Model directory in which all files are stored
	parser.add_argument("--model-dir", dest="model_dir", required=True, help="Directory where to read data from and where to write write logs and model parameters")
	parser.add_argument("--no-model", dest="no_model", action="store_true", help="Don't serialize models")
	# Custom file names (default options are fine for most settings)
	parser.add_argument("--training-file", dest="training_file", default="train.pkl", help="File name of training data pickle")
	parser.add_argument("--dev-file", dest="dev_file", default="dev.pkl", help="File name of dev data pickle")
	#parser.add_argument("--test-file", dest="test_file", default="test.pkl", help="File name of test data pickle")
	parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.pkl", help="File name of vocabulary pickle")
	# Model settings
	parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds (if not supplied, will be random)")
	parser.add_argument("--num-epochs", default=20, dest="num_epochs", type=int, help="Number of full passes through training set (default - 20)")
	parser.add_argument("--num-lstm-layers", default=2, dest="lstm_layers", type=int, help="Number of LSTM layers (default - 2)")
	parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers (default - 128)")
	parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate (default - 0.01)")
	parser.add_argument("--dropout", default=-1, dest="dropout", type=float, help="Amount of dropout to apply to LSTM part of graph (default - off)")
	parser.add_argument("--no-we-update", dest="no_we_update", action="store_true", help="Word Embeddings aren't updated")
	parser.add_argument("--loss-prop", dest="loss_prop", action="store_true", help="Proportional loss magnitudes")
	parser.add_argument("--use-char-rnn", dest="use_char_rnn", action="store_true", help="Use character RNN (default - off)")
	parser.add_argument("--dynet-mem", help="Ignore this external argument")
	parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
	options = parser.parse_args()
	
	# create folder if required
	if not os.path.exists(options.model_dir):
		os.mkdir(options.model_dir)

	# ===-----------------------------------------------------------------------===
	# Set up logging
	# ===-----------------------------------------------------------------------===
	logging.basicConfig(filename=options.model_dir + "/training.log", filemode="w", format="%(message)s", level=logging.INFO)
	train_dev_cost = csv.writer(open(options.model_dir + "/train_dev_loss.csv", 'w'))
	train_dev_cost.writerow(["Train.cost", "Dev.cost"])


	# ===-----------------------------------------------------------------------===
	# Log run parameters
	# ===-----------------------------------------------------------------------===

	logging.info(
	"""
	Model directory: {}
	Pretrained Embeddings: {}
	Num Epochs: {}
	LSTM: {} layers, {} hidden dim
	Concatenating character LSTM: {}
	Initial Learning Rate: {}
	Dropout: {}
	LSTM loss weights proportional to attribute frequency: {}

	""".format(options.model_dir, options.word_embeddings, options.num_epochs, options.lstm_layers, options.hidden_dim, options.use_char_rnn, options.learning_rate, options.dropout, options.loss_prop))

	if options.debug:
		print("DEBUG MODE")

	# ===-----------------------------------------------------------------------===
	# Read in dataset
	# ===-----------------------------------------------------------------------===
	
	training_instances = pickle.load(open(options.model_dir + "/" + options.training_file, "rb"))
	dev_instances = pickle.load(open(options.model_dir + "/" + options.dev_file, "rb"))
	vocab_dataset = pickle.load(open(options.model_dir + "/" + options.vocab_file, "rb"))
	training_vocab = vocab_dataset["training_vocab"]
	w2i = vocab_dataset["w2i"]
	t2is = vocab_dataset["t2is"]
	c2i = vocab_dataset["c2i"]
	i2w = { i: w for w, i in w2i.items() } # Inverse mapping
	i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }
	i2c = { i: c for c, i in c2i.items() }

	# ===-----------------------------------------------------------------------===
	# Build model and trainer
	# ===-----------------------------------------------------------------------===
	if options.word_embeddings is not None:
		word_embeddings = utils.read_pretrained_embeddings(options.word_embeddings, w2i)
	else:
		word_embeddings = None

	tagset_sizes = { att: len(t2i) for att, t2i in t2is.items() }

	if options.loss_prop:
		att_props = get_att_prop(training_instances)
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
					   save_settings_to_file=None if options.no_model else "model_settings.pkl")

	trainer = dy.MomentumSGDTrainer(model.model, options.learning_rate, 0.9)
	logging.info("Training Algorithm: {}".format(type(trainer)))

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
		logging.info("\n")
		logging.info("Epoch {} complete".format(epoch + 1))
		logging.info("Number of training instances: {}".format(len(training_instances)))
		logging.info("Training Loss: {}".format(train_loss))
		logging.info("")
		# here used to be a learning rate update, no longer supported in dynet 2.0
		#print(trainer.status())

		# evaluate dev data
		if options.debug:
			d_instances = dev_instances[0:int(len(dev_instances)/10)]
		else:
			d_instances = dev_instances
			
		if epoch+1 == options.num_epochs:
			new_devout_file_name = "{}/devout_final.txt".format(options.model_dir)
		else:
			new_devout_file_name = "{}/devout_epoch{:02d}.txt".format(options.model_dir, epoch + 1)
		
		logging.info("Evaluate dev data")
		dev_loss = evaluate(model, d_instances, new_devout_file_name, t2is, i2ts, i2w, i2c, training_vocab)
		logging.info("Dev Loss: {}".format(dev_loss))
		train_dev_cost.writerow([train_loss, dev_loss])

		if epoch > 1 and epoch % 10 != 0: # leave outputs from epochs 1,10,20, etc.
			old_devout_file_name = "{}/devout_epoch{:02d}.txt".format(options.model_dir, epoch)
			os.remove(old_devout_file_name)

		# serialize model
		if not options.no_model:
			if epoch+1 == options.num_epochs:
				new_model_file_name = "{}/model_final.bin".format(options.model_dir)
			else:
				new_model_file_name = "{}/model_epoch{:02d}.bin".format(options.model_dir, epoch + 1)
			logging.info("")
			logging.info("Saving model to {}".format(new_model_file_name))
			model.save(new_model_file_name)
			if epoch > 1 and epoch % 10 != 0: # leave models from epochs 1,10,20, etc.
				logging.info("Removing files from previous epoch.")
				old_model_file_name = "{}/model_epoch{:02d}.bin".format(options.model_dir, epoch)
				os.remove(old_model_file_name)
		# epoch loop ends
