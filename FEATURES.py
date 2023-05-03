
CATEGORICAL_STRING_FEATURES = [
'Contract',
'Dependents',	
'DeviceProtection'	,
'InternetService',
'MultipleLines'	,
'OnlineBackup',	
'OnlineSecurity'	,
'PaperlessBilling',
'Partner'	,
'PaymentMethod'	,
'PhoneService',	
'StreamingMovies'	,
'StreamingTV'	,
'TechSupport',
'gender'
]


DENSE_FLOAT_FEATURE_KEYS =[
"MonthlyCharges",
"TotalCharges","tenure"
]

rest= ["SeniorCitizen"]

LABEL_KEY="Churn"

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000
# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 100
# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

def t_name(key):
  """
  Rename the feature keys so that they don't clash with the raw keys when
  running the Evaluator component.
  Args:
    key: The original feature key
  Returns:
    key with '_xf' appended
  """
  return key + '_xf'
