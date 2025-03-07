import sklearn
import os
from sklearn.model_selection import train_test_split

cvs_path = r'D:\hqy\datasets\IQA\LIVE\LIVE.csv'
train_txt_save_path = r'D:\hqy\projects\PythonProjects\MFF_IQA\datasets\LIVE\LIVE_train333.txt'
test_txt_save_path = r'D:\hqy\projects\PythonProjects\MFF_IQA\datasets\LIVE\LIVE_test333.txt'

datas = []

with open(cvs_path, 'r') as listFile:
    for i, line in enumerate(listFile):
        if i == 0:
            continue

        ref, dis, score = line[:-1].split(',')[0:3]
        datas.append([dis, dis, score])

train_list, test_list = train_test_split(datas, test_size=0.2, random_state=333)

with open(train_txt_save_path, 'w') as f:
    for item in train_list:
        f.writelines(','.join(item) + '\n')
    f.close()

with open(test_txt_save_path, 'w') as f:
    for item in test_list:
        f.writelines(','.join(item) + '\n')
    f.close()