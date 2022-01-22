# encoder and decoder based language tranlation using attention
# web source: https://trungtran.io/2019/03/29/neural-machine-translation-with-attention-mechanism/
#%%
import tensorflow as tf
from tensorflow.python.eager.context import context_safe
from tensorflow.python.keras.api import keras
import numpy as np 
from tensorflow.python.keras import layers, optimizer_v2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.ops.gen_linalg_ops import batch_self_adjoint_eig
from tensorflow.python.ops.random_ops import categorical
from tensorflow.python.ops.variable_scope import variable_creator_scope
import tensorflow_text as tf_text
import typing 
from typing import Any, Tuple 
import pathlib
import pandas as pd
import os, unicodedata, re
#%%
keras.backend.clear_session()
sample_size = 10000
max_vocab_size = 2000
NUM_EPOCHS = 50
BATCH_SIZE = 5
#%%
raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),
    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
)
#%%
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
# %%
def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

# %%
raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)
raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start>' + ' ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' ' + '<end>' for data in raw_data_fr]
en_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = keras.preprocessing.sequence.pad_sequences(data_en, padding='post')
fr_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = keras.preprocessing.sequence.pad_sequences(data_fr_in, padding='post')
data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = keras.preprocessing.sequence.pad_sequences(data_fr_out, padding='post')
# %%
dataset = tf.data.Dataset.from_tensor_slices(
    (data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(len(data_en)).batch(BATCH_SIZE)
#%%
class LA(keras.layers.Layer):
    def __init__(self, rnn_size):
        super(LA, self).__init__()
        self.wa = layers.Dense(rnn_size)
    def call(self, decoder_output, encoder_output):
        # Dot score: h_t Wa h_s
        # encoder_output: (b, s, d)
        # decoder_output: (b, 1, d)
        # score: (b, 1, s)
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        alignment = tf.nn.softmax(score, axis=2) # (b, 1, s)
        context = tf.matmul(alignment, encoder_output)
        return context, alignment

# %%
class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c
    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))            
#%%
class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size):
        super(Decoder, self).__init__()
        self.attention = LA(rnn_size)
        self.rnn_size = rnn_size
        self.embedding = keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True)
        self.wc = keras.layers.Dense(rnn_size, activation='tanh')
        self.ws = keras.layers.Dense(vocab_size)
    def call(self, sequence, state, encoder_output):
        # sequence: (b,1) --> one word sequence 
        embed = self.embedding(sequence)
        # lstm_out: (b, 1, rnn_size)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)
        # cotext: (b, 1, rnn_size)
        # alignment: (b, 1, s)
        context, alignment = self.attention(lstm_out, encoder_output)
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)
        lstm_out = self.wc(lstm_out)
        logits = self.ws(lstm_out)
        return logits, state_h, state_c, alignment

# %%
EMBEDDING_SIZE = 128
LSTM_SIZE = 64 
en_vocab_size = len(en_tokenizer.word_index) + 1
encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
fr_vocab_size = len(fr_tokenizer.word_index) + 1
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
# source_input = tf.constant([[1,3,5,7,2,0,0,0]])
# initial_state = encoder.init_states(1)
# encoder_output, en_state_h, en_state_c = encoder(source_input, initial_state)
# target_input = tf.constant([[1,4,6,9,2,0,0]])
# decoder_output, de_state_h, de_state_c = decoder(target_input, (en_state_h, en_state_c))
# %%
def loss_func(targets, logits):
    crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)
    return loss     
# %%
optimizer = keras.optimizers.Adam()
# %%
def predict():
    test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(f'Source:: {test_source_text}')
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    # print(test_source_seq)
    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)
    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    alignments = []
    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(
            de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        # print(f"decoder_shape: out:{de_output.shape},in:{de_input.shape}")
        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])
        alignments.append(alignment.numpy())
        if out_words[-1]=='<end>' or len(out_words)==20:
            break 
    print(' '.join(out_words))   
    return np.array(alignments), test_source_text.split(' '), out_words 
#%%
# training step --
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(1)):
        loss = 0.0 
        with tf.GradientTape() as tape:
            en_outputs = encoder(source_seq, en_initial_states)
            en_states = en_outputs[1:]
            de_state_h, de_state_c = en_states
            for i in range(target_seq_out.shape[1]):
                decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
                logit, de_state_h, de_state_c, _ = decoder(
                    decoder_in, (de_state_h, de_state_c), en_outputs[0]
                )
                loss+=loss_func(target_seq_out[:,i], logit)
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    return loss / target_seq_out.shape[1]
# %%
for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in, target_seq_out, en_initial_states)
    print(f'Epoch {e+1} Loss: {loss.numpy():.4f}')
    try:
        predict()
    except Exception:
        continue

