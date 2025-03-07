import sklearn
import os
from sklearn.model_selection import train_test_split

seed = 333
root_dir = r'D:\hqy\projects\PythonProjects\DR_IQA\datasets'
datasets = ['LIVE', 'CSIQ', 'TID_2013', 'KADID10K', 'PIPAL']
n = len(datasets)

if os.path.exists(root_dir) == False:
    os.mkdir(root_dir)
for dataset in datasets:
    if os.path.exists(os.path.join(root_dir, dataset)) == False:
        os.mkdir(os.path.join(root_dir, dataset))

csv_paths = [
    r'D:\hqy\datasets\IQA\LIVE\LIVE.csv',
    r'D:\hqy\datasets\IQA\CSIQ\CSIQ.csv',
    r'D:\hqy\datasets\IQA\TID_2013\TID_2013.csv',
    r'D:\hqy\datasets\IQA\KADID10K\KADID10K.csv',
    r'D:\hqy\datasets\IQA\PIPAL\PIPAL.csv'
]
dis_paths = [
    r'D:\hqy\datasets\IQA\LIVE\distorted_images',
    r'D:\hqy\datasets\IQA\CSIQ\distorted_images',
    r'D:\hqy\datasets\IQA\TID_2013\distorted_images',
    r'D:\hqy\datasets\IQA\KADID10K\distorted_images',
    r'D:\hqy\datasets\IQA\PIPAL\distorted_images'
]

for index in range(4):
    csv_path = csv_paths[index]
    dis_path = dis_paths[index]
    names = []

    if datasets[index] == 'LIVE':
        type_list = os.listdir(os.path.join(dis_path))
        for type in type_list:
            for name in os.listdir(os.path.join(dis_path, type)):
                names.append(type + "/" + name)
    else:
        names = os.listdir(dis_path)

    train_names, test_names = train_test_split(names, test_size=0.2, random_state=seed)
    train_names = set(train_names)
    test_names = set(test_names)

    train_list = []
    test_list = []
    with open(csv_path, 'r') as listFile:
        for i, line in enumerate(listFile):
            if i == 0:
                continue

            ref, dis, score = line[:-1].split(',')[0:3]
            if dis in train_names:
                train_list.append([ref, dis, score])
            else:
                test_list.append([ref, dis, score])

    train_save_path = os.path.join(root_dir, datasets[index], datasets[index] + '_train' + str(seed) + '.txt')
    test_save_path = os.path.join(root_dir, datasets[index], datasets[index] + '_test' + str(seed) + '.txt')

    with open(train_save_path, 'w') as f:
        for item in train_list:
            f.writelines(','.join(item) + '\n')
        f.close()

    with open(test_save_path, 'w') as f:
        for item in test_list:
            f.writelines(','.join(item) + '\n')
        f.close()