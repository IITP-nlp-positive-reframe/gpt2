# GPT2 for positive reframing

This repository contains reimplementation of GPT2 model suggested in *Inducing Positive Perspectives with Text Reframing*.

### Dataset
Data used to train the model is under ```data``` directory. Among many data files, ```train.csv```, ```dev.csv```, ```test.csv``` are used.

### How to run Training
To train the model, run below command.  
```
python gpt_2_fine_tuning_w_hugging_face_&_pytorch.py
```

### How to run Inference
```
python gpt2_generate.py
```
It generates test prediction of the trained model in csv format. 
Load your trained model in this line of code ```model.load_state_dict(torch.load('your_trained_model_path/pytorch_model.bin'))```


### How to run Evaluation
```
python eval.py --file "generation_res.csv"
```
It reads xlsx or csv files generated from the inferene stage. It gives you BLEU, Rouge, BertScore, and the difference in Text blob sentiment scores, as in the original paper.
The input file contains ```input```, ```output```, ```gt```(gt stands for ground truth). 

### Sample prediction file
The test prediction result is provided in ```generation_res.csv```
