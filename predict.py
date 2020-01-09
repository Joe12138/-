from fast_bert.prediction import BertClassificationPredictor
import pandas as pd
import csv
import json
import copy

predictor = BertClassificationPredictor(model_path='./Data/labor/data/model/keda/model_out',
                                        label_path='./Data/labor/new_data',
                                        multi_label=True,
                                        model_type='bert')

text_list = list(pd.read_csv("./Data/labor/new_data/test.csv")['text'].values)
output = predictor.predict_batch(text_list)

print(output)
