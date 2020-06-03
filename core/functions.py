import torch
import torchtext
from core import model
from core.config import ControllerConfig, MemoryConfig, TrainingConfig
import re


def load_model(path_model,device):

    #load model
    checkpoint = torch.load(path_model)
    controller_config = ControllerConfig(**checkpoint['controller_config'])
    memory_config = MemoryConfig(**checkpoint['memory_config'])
    
    len_embedding = checkpoint['len_embeddings']

    network = model.ClozeModel(controller_config, memory_config, controller_config.input_size,
     len_embedding).to(device)
    network.load_state_dict(checkpoint['state_dict'],strict=False)
    return network


def get_cloze_dataset(path):
    PATH_VOCAB = "dataset/vocab"
    TEXT = torchtext.data.Field(sequential=True, tokenize='spacy', lower=True, batch_first=True)
    vocab = torch.load(PATH_VOCAB)['vocab']
    TEXT.vocab = vocab
    LABEL = torchtext.data.LabelField(use_vocab=False)
    ID = torchtext.data.RawField()
    fields = [("id",ID), ("premise1",TEXT), ("premise2",TEXT), ("premise3",TEXT),("premise4",TEXT), ("answer1",TEXT),("answer2",TEXT),("label",LABEL)]
    dataset = torchtext.data.TabularDataset(path,format='CSV',fields=fields,skip_header=True)
    return dataset

# from https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()