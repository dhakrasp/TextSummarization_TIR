
import gensim,logging
import pickle
import numpy as np
import nltk
import json
from nltk.tokenize import RegexpTokenizer
from keras.layers import Embedding
tokenizer = RegexpTokenizer(r'\w+')
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
fname='test.txt'	#Input File
def create_embeddings(fname,model_file='Models/w2vecmodel',embed_file='embed_file.txt'):
	
	#Creates embeddinglayer
	model = gensim.models.Word2Vec.load(model_file)
	
	def readsentences(fname):
		a=[]
		with open(fname,'r') as f:
			for line in f:
				
				tokens=tokenizer.tokenize(line)
				tokens=map(lambda x: x.lower(),tokens)
				a.append(tokens)
		return a
	sentences=readsentences(fname)
	#model=gensim.models.Word2Vec(sentences,size=200,min_count=1)
	#model.save('Models/w2vecmodel')
	#print len(model['for'])
	word_vectors = model.wv
	embeddings=word_vectors.syn0
	vocab=word_vectors.vocab
	#print model.word_vec('chinese')
	embed_file=open(embed_file,'wb')
	np.save(embed_file,embeddings)
	vocabulary=dict([(k,v.index) for k,v in vocab.items()])		
	with open('data/vocabw2index.json','wb') as fp:	
		fp.write(json.dumps(vocabulary))
	embed_file.close()
def loadvocab(vocab_mapfile):				#Returns word2id and id2word 
	with open(vocab_mapfile,'rb') as f:
		word2id=json.loads(f.read())
	id2word=dict([(v, k) for k, v in word2id.items()])
	return word2id,id2word
def word2vec_Embedding(embed_file):				#KerasEmbeddingLayer
	weights = np.load(open(embed_file, 'rb'))
	layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
	return layer
	print weights

create_embeddings(fname)
word2vec_Embedding('embed_file.txt')