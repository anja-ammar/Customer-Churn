

import tensorflow as tf
import tensorflow_transform as tft
import FEATURES
from FEATURES import t_name

DENSE_FLOAT_FEATURE_KEYS=FEATURES.DENSE_FLOAT_FEATURE_KEYS
CATEGORICAL_STRING_FEATURES=FEATURES.CATEGORICAL_STRING_FEATURES
rest=FEATURES.rest
LABEL_KEY=FEATURES.LABEL_KEY
_VOCAB_SIZE=FEATURES.VOCAB_SIZE
_OOV_SIZE=FEATURES.OOV_SIZE

def preprocessing_fn(inputs):
  
  """Preprocess input columns into transformed columns."""
  # Since we are modifying some features and leaving others unchanged, we
  # start by setting `outputs` to a copy of `inputs.
  outputs ={}

  # Scale numeric columns to have range [0, 1].
  for key in DENSE_FLOAT_FEATURE_KEYS:
    outputs[t_name(key)] = tft.scale_to_z_score(
        _fill_in_missing(inputs[key]), name=key)
  

  # For all categorical columns except the label column, we generate a
  # vocabulary but do not modify the feature.  This vocabulary is instead
  # used in the trainer, by means of a feature column, to convert the feature
  # from a string to an integer id.
  for key in CATEGORICAL_STRING_FEATURES:
    # Build a vocabulary for this feature.
    outputs[t_name(key)] =  _make_one_hot(_fill_in_missing(inputs[key]), key)
  for key in rest:
    # Build a vocabulary for this feature.
    outputs[t_name(key)] =  _fill_in_missing(inputs[key])  
  outputs[LABEL_KEY]=_fill_in_missing(inputs[LABEL_KEY])

  return outputs

def _make_one_hot(x, key):
  """Make a one-hot tensor to encode categorical features.
  Args:
    X: A dense tensor
    key: A string key for the feature in the input
  Returns:
    A dense one-hot tensor as a float list
  """
  integerized = tft.compute_and_apply_vocabulary(x,
          top_k=_VOCAB_SIZE,
          num_oov_buckets=_OOV_SIZE,
          vocab_filename=key, name=key)
  depth = (
      tft.experimental.get_vocabulary_size_by_name(key) + _OOV_SIZE)
  one_hot_encoded = tf.one_hot(
      integerized,
      depth=tf.cast(depth, tf.int32),
      on_value=1.0,
      off_value=0.0)
  return tf.reshape(one_hot_encoded, [-1, depth])

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
