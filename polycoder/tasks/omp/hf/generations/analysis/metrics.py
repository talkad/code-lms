import code_bert_score
from nltk.translate.bleu_score import sentence_bleu


def calc_bleu(pred, label): 
    return sentence_bleu([label], pred)
    
def calc_code_bleu(pred, label): 
    pass

def calc_code_bert_score(pred, label): 
    return code_bert_score.score(cands=[pred], refs=[label], lang='c')[2] # f1 score