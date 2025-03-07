import os

path = r'D:\hqy\datasets\IQA_data(repaired)\KonIQ'

imageNames = os.listdir(path)

for imageName in imageNames:
    pure_name = imageName[:imageName.rfind('.')]
    new_name = pure_name + '.jpg'
    os.rename(os.path.join(path, imageName), os.path.join(path, new_name))
    print(new_name)