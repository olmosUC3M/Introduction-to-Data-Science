# -*- coding: utf-8 -*-
from six.moves import range
from six.moves.urllib.request import urlretrieve
import zipfile
import collections
import os
import tensorflow as tf
import numpy as np
import random
import string

# A function to unzip the database and compact it into a single list of characters ...
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data
  
    


class BatchGenerator(object):
    
    def __init__(self, text, batch_size=64, num_unrollings=10, vocabulary_size = 0):
        self._text = text
        self._vocabulary_size = vocabulary_size
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size #We split the text into batch_size pieces
        self._cursor = [ offset * segment for offset in range(batch_size)]    #Cursor pointing every piece
        self._last_batch = self._next_batch()
  
    #
    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0     #One hot encoding
            #print(self._text[self._cursor[b]])
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch
  
    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches
    
    
def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (mostl likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]            #Clever! The ZIP is the key function here!
    return s   

def char2id(char):
    first_letter = ord(string.ascii_lowercase[0])
    
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0
    
def id2char(dictid):
    first_letter = ord(string.ascii_lowercase[0])
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '