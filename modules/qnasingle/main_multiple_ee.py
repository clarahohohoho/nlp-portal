from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import numpy as np


def loc_main(context):

    res = []

    model_name = "./modules/qnasingle/roberta-base-squad2_model"
    tok_name = "./modules/qnasingle/roberta-base-squad2_tokenizer"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    question = 'What is the location?'

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)

    start_values, start_indices = torch.topk(torch.nn.functional.softmax(answer_start_scores, dim = 1), 3)  # Get the most likely beginning of answer with the argmax of the score
    end_values, end_indices = torch.topk(torch.nn.functional.softmax(answer_end_scores, dim = 1), 3)  # Get the most likely end of answer with the argmax of the score
    end_indices += 1

    for i in range(3):
        start = start_indices[0][i]
        end = end_indices[0][i]
        # if start < end:
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end]))
        # if end < start:
        #     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[end:start]))
        if end_values[0][i].detach().numpy() > 0.2 and answer!="" and answer != '<s>' and answer != '</s>' and '[CLS]' not in answer and '[SEP]' not in answer:
            res.append(answer.strip())

    return res

def obj_main(context):

    res = []

    model_name = "./modules/qnasingle/roberta-base-squad2_model"
    tok_name = "./modules/qnasingle/roberta-base-squad2_tokenizer"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    question = 'What is the object?'

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)

    start_values, start_indices = torch.topk(torch.nn.functional.softmax(answer_start_scores, dim = 1), 3)  # Get the most likely beginning of answer with the argmax of the score
    end_values, end_indices = torch.topk(torch.nn.functional.softmax(answer_end_scores, dim = 1), 3)  # Get the most likely end of answer with the argmax of the score
    end_indices += 1

    for i in range(3):
        start = start_indices[0][i]
        end = end_indices[0][i]
        # if start < end:
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end]))
        # if end < start:
        #     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[end:start]))
        if end_values[0][i].detach().numpy() > 0.2 and answer!="" and answer != '<s>' and answer != '</s>' and '[CLS]' not in answer and '[SEP]' not in answer:
            res.append(answer.strip())

    return res