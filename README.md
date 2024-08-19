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

# Data Description
The original dataset contained 1,369 entries. After manually removing some erroneous data, 1,325 entries remain. The images in the 1,325 entries were renumbered from 1 to 1,325.
```Shell
math_question_solution_ans.json
```
contains math questions, the corresponding image IDs, the solution process, and the reference standard answers.
```Shell
math_with_class.jsonl: 
```
This file breaks down each question into multiple sub-questions (for example, one question may contain 2-3 sub-questions). It includes the type labels for each sub-question, the knowledge point labels, the solution process for each sub-question, and the reference standard answers for each sub-question.
```Shell
question_knowledge.json
```
This file contains the image ID for each question and the corresponding knowledge point classification.

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
To evaluate performance based on specific knowledge points, run:
```Shell
python judge_by_knowledge.py
```

### 3.3 Judge by Educational Stages
To evaluate performance across different educational stages, run:
```Shell
python judge_by_stage.py
```
### 3.4 Judge by Question Types
To evaluate performance based on question types, run:
```
python judge_by_type.py
```


