# get the transition file and balance the +ve and -ve data points
import random
from IPython import embed

file_name = 'transition_annot_file_v_1.txt'
out_train_file = open(file_name[:-4] + '_balanced_train.txt','w')
out_test_file = open(file_name[:-4] + '_balanced_test.txt', 'w')

with open(file_name, 'r') as f:
    lines = f.readlines()

pos_data = []
neg_data = []

for line in lines:
    word = line[:-1].split(' ')
    if word[1] == '1':
        pos_data.append(line)
    else:
        neg_data.append(line)

factor = float(len(neg_data)) / len(pos_data)
if factor >= 2.0:
    pos_data = int(factor) * pos_data

pos_data += random.sample(pos_data, len(neg_data)- len(pos_data))

total_data = neg_data + pos_data
random.shuffle(total_data)

train_ratio = 0.9
for d in total_data:
    if random.random() <= train_ratio:
        out_train_file.write(d)
    else:
        out_test_file.write(d)

out_train_file.close()
out_test_file.close()