from datetime import datetime
startTime = datetime.now()

# scenarios = ['ni', 'multi-task-nc', 'nic']
scenarios = ['multi-task-nc']

for s in scenarios:
    !python ./naive_baseline_ted.py --scenario="{s}" --sub_dir="{s}" -cls="ResNet18" -dp=0.4
    print('time for {}: {}'.format(s, datetime.now() - startTime))
    
print('total time: {}'.format(datetime.now() - startTime))