# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# from __future__ import absolute_import, print_function, division
# from six.moves import xrange, zip
#
# import os
# import tensorflow as tf
#
# from .vocab import Vocab
#
# class MTDataset(object):
#     _PARALLEL_N = 4
#     _NULL_ID = 0
#     _BOS_ID = 1
#     _EOS_ID = 2
#     _UNK_ID = 3
#     _SHUFFLE_SEED = 3
#     _BUF_SZ = 30000
#
#     def __init__(self, src_fp, tgt_fp, src_vocab, tgt_vocab):
#         if type(src_vocab) == str:
#             assert os.path.exists(src_vocab)
#             src_vocab = Vocab(src_vocab)
#         if type(tgt_vocab) == str:
#             assert os.path.exists(tgt_vocab)
#             tgt_vocab = Vocab(tgt_vocab)
#         self._src_vocab = src_vocab
#         self._tgt_vocab = tgt_vocab
#         self._src_fp = src_fp
#         self._tgt_fp = tgt_fp
#         self._batch_size = None
#         self._maxlen = None
#
#     def _map(self, ds, func):
#         return ds.map(func, num_parallel_calls=self._PARALLEL_N).prefetch(self._BUF_SZ)
#
#     def _group_key(self, src, tgt):
#         assert self._maxlen is not None
#         bucket_n = self._maxlen // 10
#         bucket_id = tf.minimum(tf.size(tgt) // 10, bucket_n)
#         return tf.to_int64(bucket_id)
#
#     def _batch(self, _, x):
#         assert self._batch_size is not None
#         return x.padded_batch(
#             self._batch_size,
#             padded_shapes=(
#                 tf.TensorShape([None]),
#                 tf.TensorShape([None])),
#             padding_values=(
#                 self._NULL_ID,
#                 self._NULL_ID))
#
#     def get_iterator(self, batch_size=32, maxlen=50):
#         self._maxlen = maxlen
#         self._batch_size = batch_size
#         ds = tf.data.Dataset.zip((tf.data.TextLineDataset(self._src_fp), tf.data.TextLineDataset(self._tgt_fp)))
#         ds = ds.shuffle(self._BUF_SZ, self._SHUFFLE_SEED, True)
#         ds = self._map(ds, lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values))
#         ds = ds.filter(lambda src, tgt: tf.size(src) * tf.size(tgt) > 0)
#         ds = ds.filter(lambda src, tgt: tf.logical_and(tf.size(src) <= maxlen, tf.size(tgt) <= maxlen))
#         src_table = self._src_vocab.get_index_table()
#         tgt_table = self._tgt_vocab.get_index_table()
#         ds = self._map(ds, lambda src, tgt: (
#             tf.cast(src_table.lookup(src), tf.int32), tf.cast(tgt_table.lookup(tgt), tf.int32)))
#         ds = self._map(ds, lambda src, tgt: (
#             src, tf.concat([[self._BOS_ID], tgt, [self._EOS_ID]], axis=0)
#         ))
#         ds = ds.apply(tf.contrib.data.group_by_window(
#             key_func=self._group_key, reduce_func=self._batch, window_size=batch_size))
#         return ds.make_initializable_iterator(), src_table, tgt_table
