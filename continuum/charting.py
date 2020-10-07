import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def use_output_file(fn='output/20201006_rn101_lr00001_me8_wd000001'):
    with open(fn, 'r') as f:
        output = f.read()
    return output

def task_content(output, task=1):
    out_lines = output.splitlines()
    first = 0
    last = len(out_lines) - 1
    for idx, line in enumerate(output.splitlines()):
        if f"Task {task}" in line:
            first = idx
        if f"Task {task + 1}" in line:
            last = idx
            break
    return out_lines[first:last]
    
def training_accuracies(output, task=1):
    task_out_lines = task_content(output, task)
    return [float(i[19:].replace('%', '')) for i in task_out_lines if "Training accuracy: " in i]

def final_training_accuracy(output, task=1):
    return training_accuracies(output, task)[-1]


output = use_output_file()
ta = [training_accuracies(output, t) for t in range(1, 10)]
tasks = pd.DataFrame(ta).transpose()
tasks.plot()
plt.savefig('tasks.png')

    


