import sklearn
import os
from sklearn.model_selection import train_test_split

root_dir = r'D:\hqy\projects\PythonProjects\MFF_IQA\datasets'
datasets = ['CLIVE', 'CSIQ', 'KADID10K', 'KonIQ', 'LIVE2005', 'TID_2013']

os.mkdir(root_dir)
for dataset in datasets:
    os.mkdir(os.path.join(root_dir, dataset))

csv_paths = [
             r'D:\hqy\datasets\IQA\LIVE\LIVE.csv',
            ]

dis_paths = [
             r'D:\hqy\datasets\IQA\LIVE\distorted_images',
            ]

train_txt_save_paths = [
                        r'D:\hqy\projects\PythonProjects\MFF_IQA\datasets\LIVE\LIVE_train333.txt',

                       ]

test_txt_save_paths = [
                       r'D:\hqy\projects\PythonProjects\MFF_IQA\datasets\LIVE\LIVE_test333.txt',

                      ]

total_len = len(csv_paths)
for index in range(total_len):
    names = os.listdir(dis_paths[index])
    train_names, test_names = train_test_split(names, test_size=0.2, random_state=333)

    train_names = set(train_names)
    test_names = set(test_names)

    train_list = []
    test_list = []
    with open(csv_paths[index], 'r') as listFile:
        # for line in listFile:
        for i, line in enumerate(listFile):
            if i == 0:
                continue

            ref, dis, score = line[:-1].split(',')[0:3]
            if dis in train_names:
                train_list.append([dis, dis, score])
            else:
                test_list.append([dis, dis, score])

    with open(train_txt_save_paths[index], 'w') as f:
        for item in train_list:
            f.writelines(','.join(item) + '\n')
        f.close()

    with open(test_txt_save_paths[index], 'w') as f:
        for item in test_list:
            f.writelines(','.join(item) + '\n')
        f.close()