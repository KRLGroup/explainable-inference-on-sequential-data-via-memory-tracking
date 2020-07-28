import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import absl.flags
import absl.app
import pickle
import yaml
import numpy as np
from tqdm import tqdm
from core import model
import core.dnc.explanation
from core import functions
from core.config import ControllerConfig, MemoryConfig, TrainingConfig

# user flags
FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_string("path_training", None, "Path where is stored the csv dataset")
absl.flags.DEFINE_string("path_val", None, "Path where is stored the csv dataset")
absl.flags.DEFINE_integer("top_k", 25, "Number of read cells considered for each step")
absl.flags.DEFINE_boolean("use_surrogate", False, " Whether to extract surrogate ground truth for explanation")

absl.flags.mark_flag_as_required("path_model")
absl.flags.mark_flag_as_required("path_training")
absl.flags.mark_flag_as_required("path_val")


def run_explanations(network, explanation_module, data_iterator):
    network.eval()
    best_accuracy = 0
    worst_accuracy = 0
    best_correct = 0
    worst_correct = 0
    covered = 0
    total = 0

    #print stuff    
    pbar = tqdm()
    pbar.reset(total=len(data_iterator))

    for _, data in enumerate(data_iterator):       
        
        (_, p1, p2, p3, p4, a1, a2), y = data

        y = y - 1 # gold index
        story = torch.cat((p1,p2,p3,p4),1)
        background = [p1,p2,p3,p4]
        answers = [a1,a2]
        total += y.size(0)
        #get output        
        with torch.no_grad():
            outcome, rh, wh = network(story,[a1,a2])
            predicted = torch.argmax(outcome, 1)

            for index_elem in range(p1.shape[0]):
                elem_background = [p1[index_elem:index_elem+1,:], p2[index_elem:index_elem+1,:],p3[index_elem:index_elem+1,:],p4[index_elem:index_elem+1,:]]
                elem_answers = [a1[index_elem:index_elem+1,:], a2[index_elem:index_elem+1,:]]
                elem_predicted = predicted[index_elem]
                sgt = explanation_module.get_sgt(network, elem_background,elem_answers )
                
                # case where there are contraddictory surrogate ground truth
                if len(set(sgt)) > 1:
                    covered += 1
                    rank, _ = explanation_module.get_rank(elem_background,wh[0][0],rh[elem_predicted.item()+1][0] )
                    best_prediction = sgt[rank[0]-1]
                    best_correct += (elem_predicted == best_prediction).sum().item()
                    worst_prediction = sgt[rank[-1]-1]
                    worst_correct += (elem_predicted == worst_prediction).sum().item()
                    best_accuracy = float(best_correct / covered) if best_correct > 0 else 0
                    worst_accuracy = float(worst_correct / covered) if worst_correct > 0 else 0
        #print
        pbar.set_postfix({'Best':best_accuracy,'Worst':worst_accuracy, 
        'cov':covered/total})
        pbar.update()

    pbar.close()
    return best_accuracy, worst_accuracy

def run_training_epoch(network, data_iterator, loss_function, optimizer, max_grad_norm):
    network.train()

    # init cumulative variables
    accuracy = 0
    correct = 0
    total = 0
    losses = []

    # print utility
    pbar = tqdm()
    pbar.reset(total=len(data_iterator))

    #data_iterator.init_epoch()

    for _, data in enumerate(data_iterator):       
        
        optimizer.zero_grad()
        (_, p1, p2, p3, p4, a1, a2), y = data
        y = y - 1 # gold index
        story = torch.cat((p1,p2,p3,p4),1)

        # get output
        outcome, _, _ = network(story,[a1,a2])
        predicted = torch.argmax(outcome, 1)

        # get loss
        loss = loss_function(outcome,y)
        loss.backward()
        losses.append(loss.item())
        
        # update metrics
        correct += (predicted == y).sum().item()
        total += y.size(0)
        accuracy = float(correct / total) if correct > 0 else 0  
 
        # update weights
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        pbar.set_postfix({'Acc':accuracy})
        #print
        pbar.update()

    pbar.close()
    return accuracy, np.mean(losses)

def run_val_epoch(network, data_iterator):
    network.eval()

    accuracy = 0
    correct = 0
    total = 0
    pbar = tqdm()
    pbar.reset(total=len(data_iterator))

    with torch.no_grad():
        for _, data in enumerate(data_iterator):
            (_,p1, p2, p3, p4, a1, a2), y = data

            y = y - 1 # gold index
            story = torch.cat((p1,p2,p3,p4),1)


            outcome, _, _ = network(story,[a1,a2])
            # update metrics
            predicted = torch.argmax(outcome, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            accuracy = float(correct / total) if correct > 0 else 0  

            
            #print
            pbar.set_postfix({'Acc':accuracy})
            pbar.update()
    pbar.close()
    return accuracy

def run_training(path_training, path_val, path_model, top_k, required_explanation):
    #get configuration from dict and user
    config_dict = yaml.safe_load(open("config.yaml", 'r'))
    controller_config = ControllerConfig(**config_dict['controller'])
    memory_config = MemoryConfig(**config_dict['memory'])
    training_parameters = TrainingConfig(**config_dict['training'])

    # get available device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = functions.get_cloze_dataset(path_training)
    val_dataset = functions.get_cloze_dataset(path_val)

    train_iterator = torchtext.data.Iterator(train_dataset,batch_size=training_parameters.batch_size, train=True, shuffle=True, device=DEVICE)
    val_iterator = torchtext.data.Iterator(val_dataset,batch_size=training_parameters.batch_size, train=False, sort=False,device=DEVICE)

    # Get Embedding
    vocab = torch.load("dataset/vocab")['vocab']
    embedding_pretrained_weights = vocab.vectors
    pre_trained_embeddings = torch.as_tensor(embedding_pretrained_weights).to(DEVICE)
    padding_index=1
    embedding_dim = len(embedding_pretrained_weights[0])


    #init model
    network = model.ClozeModel(controller_config, memory_config, embedding_dim,len(pre_trained_embeddings),dropout=training_parameters.dropout).to(DEVICE)
    network.embeddings.weight.data.copy_(pre_trained_embeddings)
    network.embeddings.weight.requires_grad = True
    
    explanation_mod = core.dnc.explanation.ExplanationModule(padding_value=padding_index,top_k=top_k)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=training_parameters.learning_rate, eps=1e-7)

    # initialize variables
    top1_acc = 0.0
    for epoch in range(1,11):
        print("Running epoch {}".format(epoch))
        _,_ = run_training_epoch(network,train_iterator,loss_function,optimizer,training_parameters.max_grad_norm)
        print("Validation epoch {}".format(epoch))
        accuracy = run_val_epoch(network,val_iterator)
        if required_explanation:
            print("Explaining training dataset")
            run_explanations(network,explanation_mod,train_iterator)
            print("Explain validation dataset")
            run_explanations(network,explanation_mod,val_iterator)

        if accuracy > top1_acc:
            top1_acc = accuracy
            print("saving model...")
            checkpoint = {'controller_config':config_dict['controller'], 'memory_config':config_dict['memory'],
            'state_dict':network.state_dict(), 'len_embeddings':len(pre_trained_embeddings)}
            torch.save(checkpoint, path_model)

def main(argv):
    path_model = FLAGS.path_model
    path_training = FLAGS.path_training
    path_val = FLAGS.path_val
    top_k = FLAGS.top_k
    use_surr = FLAGS.use_surrogate
    run_training(path_training,path_val, path_model, top_k, use_surr)
    print("Training process ended! The new model is stored on {}.".format(path_model))

if __name__ == '__main__':
  absl.app.run(main)