import torch
import torchtext
import absl.flags
import absl.app
from tqdm import tqdm
from core.dnc.explanation import ExplanationModule
from core import functions

# user flags
FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("model", None, "Path of the trained model")
absl.flags.DEFINE_string("dataset", None, "Path where is stored the csv dataset")
absl.flags.DEFINE_integer("top_k", 25, "Number of read cells considered for each step")

absl.flags.mark_flag_as_required('model')
absl.flags.mark_flag_as_required('dataset')

def run_explanations(network, explanation_module, data_iterator):
    network.eval()

    best_accuracy = 0
    worst_accuracy = 0
    best_correct = 0
    worst_correct = 0

    covered = 0
    total = 0
    
    accuracy = 0
    correct = 0
    #print stuff    
    pbar = tqdm()
    pbar.reset(total=len(data_iterator))

    for _, data in enumerate(data_iterator):       
        
        (_,p1, p2, p3, p4, a1, a2), y = data
        y = y - 1 # gold index
        story = torch.cat((p1,p2,p3,p4),1)
        background = [p1,p2,p3,p4]
        answers = [a1,a2]

        #get output        
        outcome, rh, wh = network(story,[a1,a2])
        predicted = torch.argmax(outcome, 1)


        #update stats
        correct += (predicted == y).sum().item()
        total += y.size(0)
        accuracy = float(correct / total) if correct > 0 else 0  

        with torch.no_grad():
            sgt = explanation_module.get_sgt(network, background,answers )

            # case where there are contraddictory surrogate ground truth
            if len(set(sgt)) > 1:
                covered += 1
                rank, _ = explanation_module.get_rank(network, background,wh[0][0],rh[predicted.item()+1][0] )
                best_prediction = sgt[rank[0]-1]
                best_correct += (predicted == best_prediction).sum().item()
                worst_prediction = sgt[rank[-1]-1]
                worst_correct += (predicted == worst_prediction).sum().item()
                best_accuracy = float(best_correct / covered) if best_correct > 0 else 0
                worst_accuracy = float(worst_correct / covered) if worst_correct > 0 else 0
        #print
        pbar.set_postfix({'Accuracy':accuracy, 'Best':best_accuracy,'Worst':worst_accuracy, 
        'covered':covered/total})
        pbar.update()

    pbar.close()
    return best_accuracy, worst_accuracy

def explain(path_model, path_dataset, top_k):
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load network and explanation module
    network = functions.load_model(path_model,DEVICE)
    explanation_module = ExplanationModule(padding_value=1,top_k=top_k)
    
    # load dataset
    dataset = functions.get_cloze_dataset(path_dataset)
    iterator = torchtext.data.Iterator(dataset,batch_size=1, train=False, sort=False, device=DEVICE)

    # get explanations and print
    best, worst = run_explanations(network,explanation_module,iterator)    
    print("Best Premise Accuracy: {:.2f}".format(best))
    print("Worst Premise Accuracy: {:.2f}".format(worst))

def main(argv):

    # parsing input from command line
    path_model = FLAGS.model
    path_dataset = FLAGS.dataset
    top_k = FLAGS.top_k


    explain(path_model,path_dataset,top_k)


if __name__ == '__main__':
  absl.app.run(main)