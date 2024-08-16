# MathScape
This is the codebase for the paper MathScape: Evaluating MLLMs in multimodal Math Scenarios through a Hierarchical Benchmark.

# Environments
You can create conda environments by the following command:
```Shell
conda create -n MathScape python=3.10 
```

# MathScape Evaluation

## 1. Evaluation for Multiple models
You can change loading models in 
```Shell
model.py
```
This script handles the loading of various models, processes the questions, and outputs the results.

## 2. Evaluate Models
```Shell
eval.py
```
This script compares the problem-solving process and the standard answers as references, evaluates the results output by the models, and writes the evaluation results into a `jsonl` file.

## 3. Judge Model Performances
After Evaluating all the models and get the jsonl files, you can judge the model's performance.
### 3.1 Judge all the Performances
```Shell
judge_all.py
```
This script directly judges whether each question is answered correctly or not, calculates the overall average accuracy of each model, and includes functions like `judge_by_xx` to read the evaluation results from the `jsonl` file. It evaluates the accuracy of each model based on different dimensions such as knowledge points, stages, and question types.
### 3.1 Judge by Knowledge Points
By running the following you can judge the model by Knowledge Points.
```Shell
python judge_by_knowledge.py
```
### 3.2 Judge by Educational Stages
By running the following you can judge the model by Educational Stages.
```Shell
python judge_by_stage.py
```
### 3.3 Judge by Question Types

By running the following you can judge the model by Question Types.
```
python judge_by_type.py
```


