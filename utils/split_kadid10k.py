import os
import shutil

csv_path = r'D:\hqy\datasets\IQA\KonIQ\KonIQ.csv'
path = r'D:\hqy\datasets\IQA\KonIQ\distorted_images'
remove_path = r'D:\hqy\datasets\IQA\KonIQ\other300'

names = set()
with open(csv_path, 'r') as listFile:
    # for line in listFile:
    for i, line in enumerate(listFile):
        dis, score = line[:-1].split(',')[0:3]
        names.add(dis)

rest = []
imageNames = os.listdir(path)
for name in imageNames:
    if name not in names:
        shutil.move(os.path.join(path, name), os.path.join(remove_path, name))
