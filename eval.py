import torch
import torchtext
import absl.flags
import absl.app
from tqdm import tqdm
from core import functions

# user flags
FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("model", None, "Path of the trained model")
absl.flags.DEFINE_string("dataset", None, "Path where is stored the csv dataset")
absl.flags.mark_flag_as_required('model')
absl.flags.mark_flag_as_required('dataset')


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

            pbar.set_postfix({'Acc':accuracy})
            pbar.update()
    pbar.close()
    return accuracy

def evaluate(path_model, path_dataset):
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load network and explanation module
    network = functions.load_model(path_model,DEVICE)
    
    # load dataset
    dataset = functions.get_cloze_dataset(path_dataset)
    iterator = torchtext.data.Iterator(dataset,batch_size=1, train=False, sort=False, device=DEVICE)
    
    # get accuracy
    accuracy = run_val_epoch(network,iterator)    
    print("Model Accuracy: {:.3f}".format(accuracy))

def main(argv):

    path_model = FLAGS.model
    path_dataset = FLAGS.dataset
    evaluate(path_model, path_dataset)


if __name__ == '__main__':
  absl.app.run(main)