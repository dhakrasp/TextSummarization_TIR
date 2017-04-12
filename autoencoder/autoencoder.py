Class AutoEncoder():

	''' To Do:
		store the hyper params in object instance
		store paths of where to save model and its weights
		figure out how to use pretrained embeddings

	'''
	def __init__(self, max_sequence_len, vocab_size, embedding_dim, rep_size, optimizer=None, word_embeddings_file=None):
		if word_embeddings_file != None:
			# Get the word embeddings from the file and set to embed_param
		else:
			embed_param = None
		self.sequence_autoencoder, _, self.encoder = self.build_model(max_sequence_len, vocab_size, embedding_dim, rep_size, optimizer, embed_param)
		self.decoder = self.build_decoder()
		self.word_embeddings = None

	def build_model(self, max_sequence_len, vocab_size, embedding_dim, rep_size, optimizer=None, embed_param=None):
		if optimizer == None:
			optimizer = optimizers.SGD(lr=1, momentum=0.95, nesterov=True)

		inputs = Input(shape=(max_sequence_len,))
		if embed_param != None:
	    	embedding_layer = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_sequence_len, weights=[embed_param], trainable=False)
	    else:
	    	embedding_layer = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_sequence_len)

	    embeddings = embedding_layer(inputs)
	    encoded = LSTM(rep_size)(inputs)
	    # encoded = LSTM(rep_size)(inputs)
	    # encoded = Dense(rep_size, activation='sigmoid')(encoded)
	    repeat = RepeatVector(max_sequence_len)(encoded)
	    outputs = LSTM(vocab_size, return_sequences=True)(repeat)
	    # soft_max = Activation('softmax')

	    sequence_autoencoder = Model(inputs, outputs)
	    # word_embeddings = Model(inputs, embeddings)
	    encoder = Model(inputs, encoded)
	    # decoder = Model(repeat, outputs)
	    # opt = optimizers.RMSprop(lr=1, rho=0.9, epsilon=1e-08)  # , clipvalue=0.5)
	    
	    sequence_autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
	    return sequence_autoencoder, embeddings, encoder

	def build_decoder(self):
		inputs = Input(shape=(rep_size,))
	    repeat = RepeatVector(max_sequence_len)(inputs)
	    lstm_layer = LSTM(vocab_size, return_sequences=True)
	    # To Do: figure out how to pass weights exactly
	    lstm_weights = self.sequence_autoencoder.layers[2].get_weights()
	    lstm_layer.set_weights = [lstm_weights]
	    outputs = lstm_layer(repeat)
	    decoder = Model(inputs, outputs)
	    # To Do: compile decoder if necessary
	    return decoder
 
	# To Do: pass num_epochs, val_split etc...
	def train(self, x, callbacks_list=None):

	    autoencoder_model, _, encoder, _ = get_model()
	    autoencoder_model.summary()
	    autoencoder_model.save(model_file)

	    reducelr_cb = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-20)
	    checkpoint_cb = ModelCheckpoint(model_weights, period=5)
	    earlystopping_cb = EarlyStopping(min_delta=0.0001, patience=10)
	    callbacks_list = [reducelr_cb, checkpoint_cb]

		y = np.array([to_categorical(sample - 1, num_classes=vocab_size) for sample in x], dtype=np.int32)
		autoencoder_model.fit(x, y, batch_size=batch_size, verbose=1, epochs=num_epochs, validation_split=val_split, callbacks=callbacks_list)

	def predict(self, x):
		return self.sequence_autoencoder.predict(x)

	def encode(self, sentence):
		return self.encoder.predict(sentence)

	def decode(self, sentence_vector):
		return self.decoder(sentence_vector)
