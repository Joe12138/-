from django.shortcuts import render
from fast_bert.prediction import BertClassificationPredictor
from pyltp import SentenceSplitter
import json
from predict import BERTModel

thresholds = {"divorce": [0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.4, 0.3, 0.3],
              "labor": [0.5, 0.4, 0.4, 0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.2, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3], 
              "loan": [0.4, 0.4, 0.4, 0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.3]}

def build_tags_dict(case_type:str):
    if case_type == 'divorce':
        label_path = '/home/zf/lyy/rpj/cail2019_track2/data/divorce/tags.txt'
        tag_path = '/home/zf/lyy/rpj/cail2019_track2/data/divorce/selectedtags.txt'
    elif case_type == 'loan':
        label_path = '/home/zf/lyy/rpj/cail2019_track2/data/loan/tags.txt'
        tag_path = '/home/zf/lyy/rpj/cail2019_track2/data/loan/selectedtags.txt'
    elif case_type == 'labor':
        label_path = '/home/zf/lyy/rpj/cail2019_track2/data/labor/tags.txt'
        tag_path = '/home/zf/lyy/rpj/cail2019_track2/data/labor/selectedtags.txt'
    tag_dict={}
    with open(label_path,'r') as labels,open(tag_path,'r') as tags:
        for label in labels:
            tag_dict[label[:-1]] = tags.readline()[:-1]
    return tag_dict
    
def prediction(text_list:list, case_type:str):
    if case_type == 'divorce':
        model_path = '/home/zf/lyy/Data/divorce/data/model/kedaV3/model_out'
        label_path = '/home/zf/lyy/Data/divorce/data/new_data'
    elif case_type == 'loan':
        model_path = '/home/zf/lyy/Data/loan/data/model/keda/model_out'
        label_path = '/home/zf/lyy/Data/loan/new_data'
    elif case_type == 'labor':
        model_path = '/home/zf/lyy/Data/labor/data/model/keda/model_out'
        label_path = '/home/zf/lyy/Data/labor/new_data'
    else:
        raise Exception('No this type')
    tag_dict = build_tags_dict(case_type)
    predictor = BertClassificationPredictor(model_path=model_path,
                                            label_path=label_path,
                                            multi_label=True,
                                            model_type='bert')
    output = predictor.predict_batch(text_list)

    detail_dic = {}
    result_dic = {}
    for i in range(len(output)):
        lab_list = []
        for key in output[i]:
            if float(key[1]) > 0.5:
                if key[0] not in result_dic.keys():
                    result_dic[key[0]] = 1
                else:
                    result_dic[key[0]] += 1
                if '21' not in key[0]:
                    lab_list.append(tag_dict[key[0]])
        detail_dic[text_list[i]] = lab_list   
    final_result_dic = {}
    for key in result_dic.keys():
        if '21' not in key:
            final_result_dic[tag_dict[key]] = result_dic[key]
    return final_result_dic, detail_dic

def prediction2(text_list:list,case_type:str):
    if case_type == 'divorce':
        model_path = '/home/zf/lyy/rpj/cail2019_track2/pb/divorce/model.pb'
        label_path = '/home/zf/lyy/rpj/cail2019_track2/data/divorce/tags.txt'
    elif case_type == 'loan':
        model_path = '/home/zf/lyy/rpj/cail2019_track2/pb/loan/model.pb'
        label_path = '/home/zf/lyy/rpj/cail2019_track2/data/loan/tags.txt'
    elif case_type == 'labor':
        model_path = '/home/zf/lyy/rpj/cail2019_track2/pb/divorce/model.pb'
        label_path = '/home/zf/lyy/rpj/cail2019_track2/data/divorce/tags.txt'
    else:
        raise Exception('No this type')
    tag_dict = build_tags_dict(case_type)
    model_1 = BERTModel(task=case_type, pb_model=model_path,
                        tagDir=label_path, threshold=thresholds[case_type],
                        vocab_file="/home/zf/lyy/rpj/cail2019_track2/chinese_L-12_H-768_A-12/vocab.txt")
    predicts_1 = model_1.getAllResult(text_list)
    print(predicts_1)
    
    detail_dic = {}
    labels_dic = {}
    for index, predict in enumerate(predicts_1):
        labels = []
        for label in predict:
            labels.append(tag_dict[label])
            if label not in labels_dic.keys():
                labels_dic[label] = 1
            else:
                labels_dic[label] += 1
        detail_dic[text_list[index]] = labels
    result={}
    for res in labels_dic.keys():
        result[tag_dict[res]]=labels_dic[res]
    return result, detail_dic

def text2list(text:str):
    sents = SentenceSplitter.split(text)
    for sentence in sents:
        print(sentence)
    return sents

def hello(request):
    return render(request, 'start.html')

def get_text(request):
    request_data = {}
    if request.POST:
        request_data['text'] = request.POST['text']
        request_data['type'] = request.POST['type']
        request_data['model'] = request.POST['model']
    text_list = text2list(request_data['text'])
    if request_data['model'] == 'bert':
        result_dic, detail_dic = prediction(text_list, request_data['type'])
    elif request_data['model'] == 'bert_rcnn':
        result_dic, detail_dic = prediction2(text_list, request_data['type'])
    else:
        raise Exception("No this model!")
    print(result_dic)
    request_data['result'] = result_dic
    request_data['detail'] = detail_dic
    if request_data['type'] == 'divorce':
        request_data['type'] = '离婚案'
    elif request_data['type'] == 'loan':
        request_data['type'] = '借贷案'
    elif request_data['type'] == 'labor':
        request_data['type'] = '劳工案'
    else:
        raise Exception('No this type!')
    
    return render(request, 'result.html', request_data)


