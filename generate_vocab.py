import nltk
import os
import codecs
import re
import xml.etree.ElementTree as et

def generate_vocab(data_dir_name):
	word_counts = dict()
	for fname in os.listdir(data_dir_name):
		for line in codecs.open(os.path.join(data_dir_name, fname), encoding = 'utf-8'):
			tokens = nltk.word_tokenize(line)
			for t in tokens:
				tok = t.lower()
				if word_counts.has_key(tok):
					word_counts[tok] = word_counts[tok] + 1
				else:
					word_counts[tok] = 1
	return word_counts

def convert_to_raw_text(data_dir_name):
	for fname in os.listdir(data_dir_name):
		with codecs.open(os.path.join(data_dir_name, fname), encoding = 'utf-8') as f:
			xml_data = f.read()
			root = et.fromstring(xml_data)
			text = root.find('content').text
			re.sub('<[^<]+?>', '', text)
			with codecs.open(os.path.join(data_dir_name, fname), mode = 'w', encoding = 'utf-8') as raw:
				raw.write(text)
			

if __name__ == '__main__':
	data_dir_name = 'raw_data'
	vocab_file_name = 'raw_data/vocab'
	word_counts = generate_vocab(data_dir_name)
	with codecs.open(vocab_file_name, mode = 'a', encoding = 'utf-8') as f:
		for k,v in word_counts.items():
			f.write(u'{}\t{}\n'.format(k, v))