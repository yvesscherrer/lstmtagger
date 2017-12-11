# what do we need for testing?
# test data + index mappings (stored in model, or in dataset?)
# model + parameters (is everything restored?)
# evaluation procedure as below

# todo inside model:
# save_settings(): pickle of a settings dict that allows us to recreate an LSTMTagger object and that contains the lookup dicts; this procedure is called once at the start of the training process
# save_parameters(): rename save(); this is created after each n epochs (as now) and allows to populate the parameters within the LSTMTagger created from save_settings()

'''
Main application script for tagging parts-of-speech and morphosyntactic tags. Run with --help for command line arguments.
'''
from collections import Counter, defaultdict
from evaluate_morphotags import Evaluator
from sys import maxsize
from model import LSTMTagger, get_word_chars

import collections
import argparse
import random
import pickle
import logging
import progressbar
import os
import dynet as dy
import numpy as np

import utils

__author__ = "Yuval Pinter and Robert Guthrie, 2017"

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
POS_KEY = "POS"
PADDING_CHAR = "<*>"

DEFAULT_WORD_EMBEDDING_SIZE = 64
DEFAULT_CHAR_EMBEDDING_SIZE = 20


if __name__ == "__main__":

	# ===-----------------------------------------------------------------------===
	# Argument parsing
	# ===-----------------------------------------------------------------------===
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", required=True, dest="dataset", help="fileid of .pkl files to use")
	parser.add_argument("--load-settings", required=True, dest="load_settings", help=".pkl file to use")
	parser.add_argument("--load-params", required=True, dest="load_params", help=".bin file to use")
	parser.add_argument("--log-dir", default="log", dest="log_dir", help="Directory where to write logs / output")
	parser.add_argument("--dynet-mem", help="Ignore this external argument")
	parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
	options = parser.parse_args()


	# ===-----------------------------------------------------------------------===
	# Set up logging
	# ===-----------------------------------------------------------------------===
	if not os.path.exists(options.log_dir):
		os.mkdir(options.log_dir)
	logging.basicConfig(filename=options.log_dir + "/testlog.txt", filemode="w", format="%(message)s", level=logging.INFO)


	# ===-----------------------------------------------------------------------===
	# Log run parameters
	# ===-----------------------------------------------------------------------===

	logging.info(
	"""
	Dataset: {}
	Settings file: {}
	Parameter file: {}

	""".format(options.dataset, options.load_settings, options.load_params))

	if options.debug:
		print("DEBUG MODE")

	# ===-----------------------------------------------------------------------===
	# Read in dataset
	# ===-----------------------------------------------------------------------===
	vocab_dataset = pickle.load(open(options.dataset + ".vocab.pkl", "rb"))
	training_vocab = vocab_dataset["training_vocab"]
	w2i = vocab_dataset["w2i"]
	t2is = vocab_dataset["t2is"]
	c2i = vocab_dataset["c2i"]
	i2w = { i: w for w, i in w2i.items() } # Inverse mapping
	i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }
	i2c = { i: c for c, i in c2i.items() }
	
	test_instances = pickle.load(open(options.dataset + ".test.pkl", "rb"))
	
	settings = pickle.load(open(options.load_settings, "rb"))
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
					   word_embedding_dim=settings["word_embedding_dim"])
	model.model.populate(options.load_params)

	# evaluate test data (once)
	logging.info("\n")
	logging.info("Number test instances: {}".format(len(test_instances)))
	model.disable_dropout()
	test_correct = Counter()
	test_total = Counter()
	test_oov_total = Counter()
	bar = progressbar.ProgressBar()
	total_wrong = Counter()
	total_wrong_oov = Counter()
	f1_eval = Evaluator(m = 'att')
	if options.debug:
		t_instances = test_instances[0:int(len(test_instances)/10)]
	else:
		t_instances = test_instances
	with open("{}/testout.txt".format(options.log_dir), 'w', encoding='utf-8') as test_writer:
		for instance in bar(t_instances):
			if len(instance.sentence) == 0: continue
			gold_tags = instance.tags
			for att in model.attributes:
				if att not in instance.tags:
					gold_tags[att] = [t2is[att][NONE_TAG]] * len(instance.sentence)
			word_chars = None if not model.use_char_rnn else get_word_chars(instance.sentence, i2w, c2i)
			out_tags_set = model.tag_sentence(instance.sentence, word_chars)

			gold_strings = utils.morphotag_strings(i2ts, gold_tags)
			obs_strings = utils.morphotag_strings(i2ts, out_tags_set)
			for g, o in zip(gold_strings, obs_strings):
				f1_eval.add_instance(utils.split_tagstring(g, has_pos=True), utils.split_tagstring(o, has_pos=True))
			for att, tags in gold_tags.items():
				out_tags = out_tags_set[att]

				oov_strings = []
				for word, gold, out in zip(instance.sentence, tags, out_tags):
					if gold == out:
						test_correct[att] += 1
					else:
						# Got the wrong tag
						total_wrong[att] += 1
						if i2w[word] not in training_vocab:
							total_wrong_oov[att] += 1

					if i2w[word] not in training_vocab:
						test_oov_total[att] += 1
						oov_strings.append("OOV")
					else:
						oov_strings.append("")

				test_total[att] += len(tags)
			test_writer.write(("\n"
							 + "\n".join(["\t".join(z) for z in zip([i2w[w] for w in instance.sentence],
																		 gold_strings, obs_strings, oov_strings)])
							 + "\n"))


	# log test results
	logging.info("POS Test Accuracy: {}".format(test_correct[POS_KEY] / test_total[POS_KEY]))
	logging.info("POS % Test OOV accuracy: {}".format((test_oov_total[POS_KEY] - total_wrong_oov[POS_KEY]) / test_oov_total[POS_KEY]))
	if total_wrong[POS_KEY] > 0:
		logging.info("POS % Test Wrong that are OOV: {}".format(total_wrong_oov[POS_KEY] / total_wrong[POS_KEY]))
	for attr in t2is.keys():
		if attr != POS_KEY:
			logging.info("{} F1: {}".format(attr, f1_eval.mic_f1(att = attr)))
	logging.info("Total attribute F1s: {} micro, {} macro, POS included = {}".format(f1_eval.mic_f1(), f1_eval.mac_f1(), False))

	logging.info("Total test tokens: {}, Total test OOV: {}, % OOV: {}".format(test_total[POS_KEY], test_oov_total[POS_KEY], test_oov_total[POS_KEY] / test_total[POS_KEY]))
