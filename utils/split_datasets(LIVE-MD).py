import sklearn
import os
from sklearn.model_selection import train_test_split

cvs_path = r'D:\hqy\datasets\IQA\LIVE_MD\LIVE_MD.csv'
ref_path = r'D:\hqy\datasets\IQA_data(repaired)\LIVE_MD\distorted_images'
train_txt_save_path = r'D:\hqy\projects\PythonProjects\MFF_IQA\datasets\LIVE_MD\LIVE_MD_train333.txt'
test_txt_save_path = r'D:\hqy\projects\PythonProjects\MFF_IQA\datasets\LIVE_MD\LIVE_MD_test333.txt'

names = os.listdir(ref_path)
train_names, test_names = train_test_split(names, test_size=0.2, random_state=333)

train_names = set(train_names)
test_names = set(test_names)

train_list = []
test_list = []
with open(cvs_path, 'r') as listFile:
    # for line in listFile:
    for i, line in enumerate(listFile):
        if i == 0:
            continue

        ref, dis, score = line[:-1].split(',')[0:3]
        if dis in train_names:
            train_list.append([dis, dis, score])
        else:
            test_list.append([dis, dis, score])

with open(train_txt_save_path, 'w') as f:
    for item in train_list:
        f.writelines(','.join(item) + '\n')
    f.close()

with open(test_txt_save_path, 'w') as f:
    for item in test_list:
        f.writelines(','.join(item) + '\n')
    f.close()