from json import load
from unittest import result
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import transformers 
from tqdm.notebook import tqdm
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences 
import matplotlib.pyplot as plt

tokenizer=transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
max_length=128
bert_model=transformers.TFBertModel.from_pretrained('bert-base-uncased')
bert_model.trainable=True

input_comment1="Thank you for understanding. I think very highly of you and would not revert without discussion."
input_comment2=" Anyone have confirmation that Sir, Alfred is no longer at the airport and is hospitalised?"
input_comment3="::: Somebody will invariably try to add Religion?  Really??  You mean, the way people have invariably kept adding """"Religion"""" to the Samuel Beckett infobox?  And why do you bother bringing up the long-dead completely non-existent """"Influences"""" issue?  You're just flailing, making up crap on the fly.::: For comparison, the only explicit acknowledgement in the entire Amos Oz article that he is personally Jewish is in the categories!"
input_comment4="i will kill you!" 

def filter_comment(comment): 
    comment=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", comment)
    return [comment]

def tokenize(data,tokenizer=tokenizer,max_length=max_length): 
    bert_outputs=[] 
    encoded_data=tokenizer.batch_encode_plus(
                    data,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_tensors="tf",
                )
        
    bert_output=bert_model(**encoded_data)
    sequence_output = bert_output.last_hidden_state
    bert_outputs.append(sequence_output)
    return bert_outputs

def get_results(result_arr):
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate','positive']
    predictions=list(result[0])
    fig=plt.figure(figsize=(10,5)) 
    plt.bar(label_cols,predictions,color="red",width=0.2)
    plt.xlabel("Comment Sense")
    plt.ylabel("value of Sense")
    plt.show()
    print(label_cols[np.argmax(result_arr)]) 


bert_op=tokenize(filter_comment(input_comment1))  
classification_model=load_model("my_custom_train_model.h5")
result=classification_model.predict(bert_op) 
get_results(result)