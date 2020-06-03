import torch
import torchtext
import absl.flags
import absl.app
from tqdm import tqdm
import model
from dnc.explanation import ExplanationModule
import functions

# user flags
FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("model", None, "Path of the trained model")
absl.flags.DEFINE_string("dataset", None, "Path where is stored the csv dataset")
absl.flags.DEFINE_integer("top_k", 25, "Number of read cells considered for each step")
absl.flags.DEFINE_integer("n_samples", 1, "Number of read cells considered for each step")
absl.flags.DEFINE_boolean("use_surrogate", False, " Whether to extract surrogate ground truth for explanation")

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
        
        (p1, p2, p3, p4, a1, a2), y = data
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

def get_sentence(dataset,id):
    for ex in dataset:
        if ex.id == id: 
            return functions.untokenize(ex.premise1) + "\n" + functions.untokenize(ex.premise2)  + "\n"\
                + functions.untokenize(ex.premise3) + "\n" + functions.untokenize(ex.premise4)  \
                + "\n" + "Ending 0. " + functions.untokenize(ex.answer1)+ \
                "\n" + "Ending 1. " + functions.untokenize(ex.answer2)

def generate_samples(n_samples,path_model, path_dataset, top_k, only_surrogate):
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load network and explanation module
    network = functions.load_model(path_model,DEVICE)
    explanation_module = ExplanationModule(padding_value=1,top_k=top_k)
    
    # load dataset
    dataset = functions.get_cloze_dataset(path_dataset)
    iterator = torchtext.data.Iterator(dataset,batch_size=1, shuffle=True, device=DEVICE)
    network.eval()
    generated = 0
    for index,data in enumerate(iterator):
        if generated >= n_samples:
            break
        (id, p1, p2, p3, p4, a1, a2), y = data
        y = y - 1 # gold index
        story = torch.cat((p1,p2,p3,p4),1)
        background = [p1,p2,p3,p4]
        answers = [a1,a2]

        # get output        
        outcome, rh, wh = network(story,[a1,a2])
        predicted = torch.argmax(outcome, 1)

        # get surrogate
        sgt = explanation_module.get_sgt(network, background,answers )
        rank, percentage = explanation_module.get_rank(network, background,wh[0][0],rh[predicted.item()+1][0] )
        if only_surrogate and len(set(sgt)) > 1 or only_surrogate is False:
            generated +=1
            print("Sentence:")
            print(get_sentence(dataset,id[0]))
            print()
            print("Predicted Answer:"+str(predicted.item()))
            print()
            for i,gt in enumerate(sgt):
                print("Using only the premise {} the model outputs: {}".format(i+1,gt))
            print()
            print("Premises rank computed by the Explanation Module:")
            for i in range(len(rank)):
                premise = rank[i]
                perc = percentage[i]
                print("Premise {} read {:3.0f}% of time".format(premise,perc))
            print("******************************************")



def main(argv):

    # parsing input from command line
    path_model = FLAGS.model
    path_dataset = FLAGS.dataset
    top_k = FLAGS.top_k
    n_samples = FLAGS.n_samples
    use_surr = FLAGS.use_surrogate

    generate_samples(n_samples, path_model,path_dataset,top_k, use_surr)


if __name__ == '__main__':
  absl.app.run(main)