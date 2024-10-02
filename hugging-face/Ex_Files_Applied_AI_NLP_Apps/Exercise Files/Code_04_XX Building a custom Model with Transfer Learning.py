"""
Training wiht Hugging Face
* use transfer learning to customize model for specific use cases
  - Use a smaller training dataset, specific to the use case
  - Freeze the encoder/decoder weights and only train the classifier/Seq2Seq
* Reuse below from Hugging Face
  - Tokenizer - to create embeddings
  - Model - based language model
  - Datasets


Build a custom model: Sentiment Analysis

    Datasets and/or customize                 Prebuilt tokenizer/customize       Checkpoint/custom
Training Text Corpus -> Label and Annotate -> Tokenize -> Vectorize           -> Build Model
(Use "Poem Sentiment" dataset from
  hugging face. Dataset already labeled)      (Prebuilt model: use checkpoint      (Use transfer learning; start with base
                                               "Distilbert-base-uncased" for the    checkpoint parameters and fine-tune)
                                               Tokenizer and Model)

if you intend to use your own dataset, you need to convert it into hugging face data format
for this conversion(Apache Arrow Format), pls find sample in ()
"""

import transformers
transformers.logging.set_verbosity_error()

"""
1. Dataset: load data set and tokenize&vectorize it
"""
from datasets import load_dataset

#Use Pretrained&prebuilt model as checkpoint from hugging face
#For Tokenizer and Model
model_name = "distilbert-base-uncased"
#Use Pre-labeled dataset from huggingface
dataset_name = "poem_sentiment"

poem_sentiments = load_dataset(dataset_name)

#Apache Arrow format
#print(poem_sentiments)
#print(poem_sentiments["test"][20:25])

#print("\nSentiment Labels used", poem_sentiments["train"].features.get("label").names)

"""
(as above data is clean and formatted, so Training Text Corpus & Label and Annotate are not required. 
will then proceed for Tokenize & Vectorize)
04.03. Proceed Tokenize & Vectorize
"""
#Encoding text
from transformers import DistilBertTokenizer
db_tokenizer = DistilBertTokenizer.from_pretrained(model_name)
def tokenize(batch):
    return db_tokenizer(batch["verse_text"], padding=True, truncation=True)

encoded_poem_sentiment = poem_sentiments.map(tokenize, batched=True, batch_size=None)
#print(encoded_poem_sentiment["train"][0:5])

#Explore input IDs and Attention Mask
print("Text: ", encoded_poem_sentiment["train"][1].get("verse_text"))
#Embedding IDs
"""
#ID start with 101(begining of the string) and end with 102(end of the string)
#in the middle of IDs corresponding to each token from "verse_text"

Length of Embedding IDs list is 28 which is much higher than the number of non-zero tokens
- this is because during tokenization, the strings are right parted with spaces, 
  up to the max length for this dataset
"""

#print("\nInput Map: ", encoded_poem_sentiment["train"][1].get("input_ids")) #Embedding IDs
#Attention mask is set to 1 for non-zero Embedding IDs
#print("\nAttention Mask : ", encoded_poem_sentiment["train"][1].get("attention_mask"))
#print("\nTotal tokens: ", len(encoded_poem_sentiment["train"][1].get("input_ids")))
#print("Non Zero tokens: ", len(list(filter(lambda x :x > 0,encoded_poem_sentiment["train"][1].get("input_ids")))))
#print("Attention = 1: ", len(list(filter(lambda x :x > 0,encoded_poem_sentiment["train"][1].get("attention_mask")))))

#Seperate training and validation sets
training_datasets = encoded_poem_sentiment["train"]
validation_datasets = encoded_poem_sentiment["validation"]

#print("\n Column Names : ", training_datasets.column_names)
#print("\n Features: ", training_datasets.features)

labels = training_datasets.features.get("label")
num_labels = len(labels.names)

"""
04.04 Creating the Model Architecture
  - involving create each layers
      - setup various hyperparameters (like activation functions and normalization)
      - initializing the weights and biases
      
when we start to transfer learning, we start with the base model: TFAutoModelForSequenceClassification
"""
from transformers import TFAutoModelForSequenceClassification


"""
2. Model Architecture


#Retreive sentiment model from the pretrained checkpoint, create it's architecture
#Load transformer checkpoint from hugging face
#this will automatically copies over the current model architecture(default will be 'architectures': ['DistilBertForMaskedLM'],),
#                                     hyperparameters
#                                     and parameters
"""
sentiment_model = (TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels))
#print(sentiment_model.get_config())

#Freeze the first layer if needed, set trainable = False
sentiment_model.layers[0].trainable = True

#pring each layer summary
# first layer is distilbert, has most of parameters , will be ignored if sentiment_model.layers[0].trainable = False
# second layer is pre_classifier
# third layer is classifier
# forth layer is dropout

#print(sentiment_model.summary())


# Customization: Add/Remove layers if needed
# sentiment_model.layers.append() / insert() / remove()

"""
Having above input 1.dataset and 2. model architecture ready
3: now we can start training customize sentiment model

Train the model with custom dataset
#Using features from a pretrained model
"""
import tensorflow as tf
batch_size = 64
tokenizer_columns = db_tokenizer.model_input_names
#import os
#os.environ['SM_FRAMEWORK'] = 'tf.keras'
#Convert to TF tensors
# Convert to TF tensors
train_dataset = training_datasets.to_tf_dataset(
    columns=tokenizer_columns, label_cols=["label"], shuffle=True,
    batch_size=batch_size)
val_dataset = validation_datasets.to_tf_dataset(
    columns=tokenizer_columns, label_cols=["label"], shuffle=False,
    batch_size=batch_size)
#setup hyperparameters
sentiment_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy())
#train model
sentiment_model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=5)

"""
04.06 Predicting Sentiment with the Custom Model

with the custom model built with transfer learning is similar to using models built from scratch
Because we're not using the pipeline, we need to do the pre-processing inputs and the outputs from the model

we will perform the post-processing output steps
"""

from datasets import Dataset, DatasetDict
infer_data = {'id':[0,1],
              'verse_text':['and be glad in the summer morning when the kindred ride on their way','how hearts were answering to his own'],
              'label':[1,0]
              }
infer_dataset = Dataset.from_dict(infer_data)
ds_dict = DatasetDict();
ds_dict["infer"] = infer_dataset
#print(ds_dict)

#Encode the dataset, similar to training  -> get Embedding IDs
enc_dataset = ds_dict.map(tokenize, batched=True, batch_size=None)

#Convert to Tensors
infer_final_dataset = enc_dataset["infer"].to_tf_dataset(columns=tokenizer_columns, shuffle=True, batch_size=batch_size)
print(infer_final_dataset)

#Predict with the model
predictions = sentiment_model.predict(infer_final_dataset)
print(predictions.logits)

import numpy as np
pred_label_ids = np.argmax(predictions.logits, axis=1)
for i in range(len(pred_label_ids)):
    print("Poem=",infer_data["verse_text"][i],
          " Predicted=", labels.names[pred_label_ids[i]],
          " Actual True-Label=", labels.names[infer_data["label"][i]])