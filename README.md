# MathScape

# Description of Scripts

## 1. model.py
This script handles the loading of various models, processes the questions, and outputs the results.

## 2. eval.py
This script compares the problem-solving process and the standard answers as references, evaluates the results output by the models, and writes the evaluation results into a `jsonl` file.

## 3. judge_all.py
This script directly judges whether each question is answered correctly or not, calculates the overall average accuracy of each model, and includes functions like `judge_by_xx` to read the evaluation results from the `jsonl` file. It evaluates the accuracy of each model based on different dimensions such as knowledge points, stages, and question types.

