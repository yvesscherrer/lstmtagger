import numpy as np

NONE_TAG = "<NONE>"
POS_KEY = "POS"


def read_pretrained_embeddings(filename, w2i):
	word_to_embed = {}
	with open(filename, "r", encoding="utf-8") as f:
		for line in f:
			split = line.split()
			if len(split) > 2:
				word = split[0]
				vec = split[1:]
				word_to_embed[word] = vec
	embedding_dim = len(word_to_embed[list(word_to_embed.keys())[0]])
	out = np.random.uniform(-0.8, 0.8, (len(w2i), embedding_dim))
	for word, embed in word_to_embed.items():
		embed_arr = np.array(embed)
		if np.linalg.norm(embed_arr) < 15.0 and word in w2i:
			# Theres a reason for this if condition.  Some tokens in ptb
			# cause numerical problems because they are long strings of the same punctuation, e.g
			# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! which end up having huge norms, since Morfessor will
			# segment it as a ton of ! and then the sum of these morpheme vectors is huge.
			out[w2i[word]] = np.array(embed)
	return out


def split_tagstring(s, uni_key=False, has_pos=False, ignore=[]):
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
			if a not in ignore:
				ret[a] = v
		else:
			if attval.split('=')[0] not in ignore:
				ret.append(attval)
	return ret


def morphotag_strings(i2ts, tag_mapping, pos_separate_col=True):
	senlen = len(list(tag_mapping.values())[0])
	key_value_strs = []

	# j iterates along sentence, as we're building the string representations
	# in the opposite orientation as the mapping
	for j in range(senlen):
		place_strs = []
		for att, seq in tag_mapping.items():
			if j >= len(seq):
				print("Accessing inexistent index: attribute {}, position {}, sequence {}".format(att, j, str(tag_mapping.items())))
				val = NONE_TAG
			else:
				val = i2ts[att][seq[j]]
			if pos_separate_col and att == POS_KEY:
				pos_str = "" if val == NONE_TAG else val
			elif val != NONE_TAG:
				place_strs.append(att + "=" + val)
		morpho_str = "|".join(sorted(place_strs))
		if pos_separate_col:
			key_value_strs.append(pos_str + "\t" + morpho_str)
		else:
			key_value_strs.append(morpho_str)
	return key_value_strs

def sortvals(dct):
	return [v for (k,v) in sorted(dct.items())]
