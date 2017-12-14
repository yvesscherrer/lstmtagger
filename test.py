'''
Main application script for loading a pre-trained tagging model and evaluating it on a test file. Run with --help for command line arguments.
'''

from model import LSTMTagger, evaluate
from make_dataset import Instance
import argparse
import pickle
import logging
import os

__author__ = "Yuval Pinter and Robert Guthrie, 2017 + Yves Scherrer"


if __name__ == "__main__":

	# ===-----------------------------------------------------------------------===
	# Argument parsing
	# ===-----------------------------------------------------------------------===
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-dir", dest="model_dir", required=True, help="Directory where to read data from and where to write write logs and model parameters")
	parser.add_argument("--settings-file", dest="settings_file", default="model_settings.pkl", help="File name of model settings pickle")
	parser.add_argument("--param-file", dest="param_file", default="model_final.bin", help="File name of parameter .bin file")
	parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.pkl", help="File name of vocabulary pickle")
	parser.add_argument("--test-file", dest="test_file", default="test.pkl", help="File name of test data pickle")
	parser.add_argument("--dynet-mem", help="Ignore this external argument")
	parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
	options = parser.parse_args()


	# ===-----------------------------------------------------------------------===
	# Set up logging
	# ===-----------------------------------------------------------------------===
	if not os.path.exists(options.model_dir):
		os.mkdir(options.model_dir)
	logging.basicConfig(filename=options.model_dir + "/test.log", filemode="w", format="%(message)s", level=logging.INFO)


	# ===-----------------------------------------------------------------------===
	# Log run parameters
	# ===-----------------------------------------------------------------------===

	logging.info(
	"""
	Model directory: {}
	Settings file: {}
	Parameter file: {}

	""".format(options.model_dir, options.settings_file, options.param_file))

	if options.debug:
		print("DEBUG MODE")

	# ===-----------------------------------------------------------------------===
	# Read in dataset
	# ===-----------------------------------------------------------------------===
	vocab_dataset = pickle.load(open(options.model_dir + "/" + options.vocab_file, "rb"))
	training_vocab = vocab_dataset["training_vocab"]
	w2i = vocab_dataset["w2i"]
	t2is = vocab_dataset["t2is"]
	c2i = vocab_dataset["c2i"]
	i2w = { i: w for w, i in w2i.items() } # Inverse mapping
	i2ts = { att: {i: t for t, i in t2i.items()} for att, t2i in t2is.items() }
	i2c = { i: c for c, i in c2i.items() }
	
	settings = pickle.load(open(options.model_dir + "/" + options.settings_file, "rb"))
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
					   populate_from_file=options.model_dir + "/" + options.param_file)

	# evaluate test data
	test_instances = pickle.load(open(options.model_dir + "/" + options.test_file, "rb"))
	logging.info("Evaluate test data")
	if options.debug:
		t_instances = test_instances[0:int(len(test_instances)/10)]
	else:
		t_instances = test_instances
	evaluate(model, t_instances, "{}/testout.txt".format(options.model_dir), t2is, i2ts, i2w, i2c, training_vocab)
