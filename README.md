# Explainable Inference On Sequential Data via MemoryTracking
Source code for the IJCAI2020 paper "Explainable Inference on Sequential Data via Memory-Tracking", Biagio La Rosa, Roberto Capobianco, Daniele Nardi. [<a href="https://www.ijcai.org/Proceedings/2020/278">LINK</a>]

**Abstract**: In this paper we present a novel mechanism to get explanations that allow to better understand network predictions when dealing with sequential data. Specifically, we adopt memory-based networks — Differential Neural Computers — to exploit their capability of storing data in memory and reusing it for inference. By tracking both the memory access at prediction time, and the information stored by the network at each step of the input sequence, we can retrieve the most relevant input steps associated to each prediction. We validate our approach (1) on a modified T-maze, which is a non-Markovian discrete control task evaluating an algorithm’s ability to correlate events far apart in history, and (2) on the Story Cloze Test, which is a commonsense reasoning framework for evaluating story understanding that requires a system to choose the correct ending to a four-sentence story. Our results show that we are able to explain agent’s decisions in (1) and to reconstruct the most relevant sentences used by the network to select the story ending in (2). Additionally, we show not only that by removing those sentences the network prediction changes, but also that the same are sufficient to reproduce the inference


# Requirements

<a href="https://pytorch.org/">PyTorch 1.5</a> <em>(architecture)</em>


<a href="https://pypi.org/project/absl-py/"> absl</a> <em>(command line input)</em>


<a href="https://pypi.org/project/tqdm/">tqdm</a> <em>(logging)</em>

<a href="https://pydantic-docs.helpmanual.io/">Pydantic</a> <em>(Configuration Management) </em>

<a href="https://pyyaml.org/wiki/PyYAMLDocumentation">PyYaml</a>  <em>(Configuration Management)</em>
# Scripts

### train.py
It trains a new model based on the configuration stored on config.yaml file.
Parameters:
   - path_model <em>(string)</em>: path and name where to store the trained model.
   - path_training <em>(string)</em>: Path where is stored the Cloze training dataset. 
   - path_val <em>(string)</em>: Path where is stored the Cloze validation dataset. 
   - use_surrogate <em>(bool: optional. Default:False)</em>: whether to get explanations during the training process. Setting True this parameters slow down the whole training process due to the calculation of surrogate ground truth (~ 5x time needed. It is needed only for the production of the paper Figure 6).
   - top_k <em>(int:optional.)</em> Threshold on the number of cells to be considered for each timestep. Needed only if use_surrogate is True.

Note: in order to change the training parameters you have to edit the file config.yaml <em>core</em>

Example:
```
python train.py --path_model=models/sample_model.pt --path_training=dataset/train.csv --path_val=dataset/val.csv
```
### explain.py
It prints, for the given dataset, the Explanation Accuracy, as defined in the paper.

Parameters:
   - model <em>(string)</em>: path where the trained model is stored.
   - dataset <em>(string)</em>: path where the dataset is stored.
   - top_k <em>(int)</em> Threshold on the number of cells to be considered for each timestep. 

<ins>Note</ins>: sometimes the accuracy of Best Premise and Worst Premise are flipped across all the datasets. We hypotize that the behavior depends on how the network learns during the training process and whether it relies on similarity or dissimilarity between the memory and the controller output. A different type of initialization of some parameters could solve this problem in future releases.

Example:
```
python explain.py --model=models/sample_model.pt --dataset=dataset/train.csv
```
### eval.py
The script prints the accuracy reached by the trained model in the given dataset.

Parameters:
   - dataset <em>(string)</em>: path where the dataset is stored.
   - model <em>(string - default:"models/sample_model.pt")</em>: path where the trained model is stored.

Example:
```
python eval.py --model=models/sample_model.pt --dataset=dataset/test.csv
```
### samples.py
The script prints some random examples from the given dataset and the relative explanations.

Parameters:
   - model <em>(string)</em>: path where the trained model is stored.
   - dataset <em>(string)</em>: path where the dataset is stored.
   - top_k <em>(int)</em> Threshold on the number of cells to be considered for each timestep. 
  - use_surrogate <em>(bool: optional. Default:False)</em>: whether to get explanations only for the example with surrogate ground truth available.

Example:
- Input
```
python samples.py --model=models/sample_model.pt --dataset=dataset/test.csv --n_samples=1
```
- Output
```
Sentence:
lenny wanted to learn to practice hypnosis.
he did a lot of research about how to do it well.
he found out that hypnosis doesn't work like he thought.
he found that hypnosis is more of a parlor trick than a power.
Ending 0. lenny found a different interest.
Ending 1. lenny became a powerful hypnotist.

Predicted Answer:1
True Answer:0

Using only the premise 1 the model outputs: 0
Using only the premise 2 the model outputs: 1
Using only the premise 3 the model outputs: 0
Using only the premise 4 the model outputs: 0

Premises rank computed by the Explanation Module:
Premise 2 read  34% of time
Premise 4 read  31% of time
Premise 3 read  28% of time
Premise 1 read   7% of time
```
### References
The porting is partially based on the code of the original DNC implementation available at https://github.com/deepmind/dnc