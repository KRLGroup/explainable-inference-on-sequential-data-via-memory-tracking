# explainable-inference-on-sequential-data-via-memory-tracking
Source code for the IJCAI2020 paper "Explainable Inference on Sequential Data via Memory-Tracking", Biagio La Rosa, Roberto Capobianco, Daniele Nardi

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
```
python samples.py --model=models/sample_model.pt --dataset=dataset/test.csv --n_samples=1
```
### References
The porting is partially based on the code of the original DNC implementation available at https://github.com/deepmind/dnc