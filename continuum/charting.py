import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def use_output_file(fn='continuum/output/20202020_rn18_rehe_sgd'):
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

def avg_accuracies(output, task=1):
    task_out_lines = task_content(output, task)
    return [float(i[18:].replace('%', '')) for i in task_out_lines if "Average Accuracy: " in i][0]


output = use_output_file()
# ta = [training_accuracies(output, t) for t in range(1, 10)]
# tasks = pd.DataFrame(ta).transpose()
avgs = [avg_accuracies(output, t) for t in range(1, 10)]
avgs = pd.DataFrame(avgs, columns=["Rehersal"])


avgs.plot(label="Rehersal")

# plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], naive_accs, '-o', label="Rehersal")
#plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], rehe_accs, '-o', label="Rehearsal")
#plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], ewc_accs, '-o', label="EWC")
plt.xlabel('Tasks Encountered', fontsize=14)
plt.ylabel('Average Accuracy', fontsize=14)
plt.title('Rehersal Strategy on Core50 w/ResNet18', fontsize=14)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.legend(prop={'size': 14})
plt.savefig('rehersal_strategy.png')

    


