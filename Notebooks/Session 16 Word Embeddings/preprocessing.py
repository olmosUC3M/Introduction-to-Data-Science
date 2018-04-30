# -*- coding: utf-8 -*-
from six.moves import range
from six.moves.urllib.request import urlretrieve
import zipfile
import collections
import os
import tensorflow as tf
import numpy as np
import random



def maybe_download(filename, expected_bytes):
    
    url = 'http://mattmahoney.net/dc/'
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(vocabulary_size,words):

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1)) ##I get it now!
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)    #To assing index to words in the vocabulary
    data = list()           #List of index words
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    #We create a dictionary with index "keys" and word values. This is used to go from index to actual words
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data, count, dictionary, reverse_dictionary

def generate_batch(data, data_index, batch_size, num_skips, skip_window):
    
    """Generate a batch of data for training.
    Args:
        batch_size: Number of samples to generate in the batch.
        
        skip_window:# How many words to consider left and right.
        
            How many words to consider around the target word, left and right.
            With skip_window=2, in the sentence above for "consider" we'll
            build the window [words, to, consider, around, the].
            
        num_skips: How many times to reuse an input to generate a label.
        
            For skip-gram, we map target word to adjacent words in the window
            around it. This parameter says how many adjacent word mappings to
            add to the batch for each target word. Naturally it can't be more
            than skip_window * 2.
            
    Returns:
        batch, labels - ndarrays with IDs.
        batch: Row vector of size batch_size containing target words.
        labels:
            Column vector of size batch_size containing a randomly selected
            adjacent word for every target word in 'batch'.
    """
    
    
    assert batch_size % num_skips == 0     #batch_size must be divisible by num_skips!
    assert num_skips <= 2 * skip_window
    batch = np.zeros(shape=(batch_size), dtype=np.int32)   #Vector!
    labels = np.zeros(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    
    # buffer is a sliding window through the 'data' list. We initially fill it
    # with the first 'span' IDs in 'data'.
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)  #Module with the total length 
    
    # Generate the batch data in two nested loops. The outer loop takes the next
    # target word from 'data'; the inner loop picks random 'num_skips' adjacent
    # words to map from the target.    
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels