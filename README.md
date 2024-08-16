# MathScape
This is the codebase for the paper MathScape: Evaluating MLLMs in multimodal Math Scenarios through a Hierarchical Benchmark.

# Environments
To set up the environment, use the following commands:
```Shell
conda create -n MathScape python=3.10
conda activate MathScape
```
Then, install the required Python packages:
```Shell
pip install -r requirements.txt
```

# MathScape Evaluation

## 1. Model Evaluation
To evaluate multiple models, you can configure the models in the model.py file. This script is responsible for loading the models, processing the questions, and generating the results.
```Shell
model.py
```

## 2. Running Evaluations
To evaluate the models' performance, run:
```Shell
eval.py
```
This script compares the model-generated solutions against the standard answers, evaluates the results, and saves the evaluation data in a jsonl file.

## 3. Assessing Model Performance
Once all models have been evaluated and the jsonl files generated, you can assess the performance of each model.
### 3.1 Judge all the Performances
To assess overall performance, run:
```Shell
judge_all.py
```
This script determines the correctness of each answer, calculates the overall average accuracy for each model, and includes functions like judge_by_xx to read and evaluate the results from the jsonl file. It also allows for evaluating accuracy based on various dimensions such as knowledge points, educational stages, and question types.
### 3.2 Judge by Knowledge Points
By running the following you can judge the model by Knowledge Points.
```Shell
python judge_by_knowledge.py
```
### 3.3 Judge by Educational Stages
By running the following you can judge the model by Educational Stages.
```Shell
python judge_by_stage.py
```
### 3.4 Judge by Question Types

By running the following you can judge the model by Question Types.
```
python judge_by_type.py
```


