import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def compute(path, save_dir):
    file_names = os.listdir(path)
    file_name_green = []
    file_name_none = []
    for file_name in tqdm(file_names):
        img = cv2.imread(os.path.join(path, file_name))
        per_image_Bmean = np.mean(img[:, :, 0])
        per_image_Gmean = np.mean(img[:, :, 1])
        per_image_Rmean = np.mean(img[:, :, 2])
        if per_image_Bmean > 65 and per_image_Gmean > 65 and per_image_Rmean > 65:
            file_name_green.append(file_name)
        else:
            file_name_none.append(file_name)
    file = open(os.path.join(save_dir, path.split('/')[-1]+'_green.txt'), 'w')

    for filename in sorted(file_name_green):
        file.write(str(filename+'\n'))
    file.close()
    file = open(os.path.join(save_dir, path.split('/')[-1]+'_normal.txt'), 'w')
    for filename in sorted(file_name_none):
        file.write(str(filename+'\n'))
    file.close()

def compute_unlabeled(path='/raid/chenby/person_rID/train/images', save_dir='/raid/chenby/person_rID/train',
                      unlabel_txt='/raid/chenby/person_rID/train/unlabel.txt'):
    with open(unlabel_txt, "r") as f:
        file_names = f.readlines()
    file_name_green = []
    file_name_none = []
    for file_name in tqdm(file_names):
        img = cv2.imread(os.path.join(path, file_name))
        per_image_Bmean = np.mean(img[:, :, 0])
        per_image_Gmean = np.mean(img[:, :, 1])
        per_image_Rmean = np.mean(img[:, :, 2])
        if per_image_Bmean > 65 and per_image_Gmean > 65 and per_image_Rmean > 65:
            file_name_green.append(file_name)
        else:
            file_name_none.append(file_name)
    file = open(os.path.join(save_dir, path.split('/')[-1]+'_green.txt'), 'w')

    for filename in sorted(file_name_green):
        file.write(str(filename+'\n'))
    file.close()
    file = open(os.path.join(save_dir, path.split('/')[-1]+'_normal.txt'), 'w')
    for filename in sorted(file_name_none):
        file.write(str(filename+'\n'))
    file.close()

def get_unlabeled_data(label_txt='/raid/chenby/person_rID/train/label.txt',
                       unlabel_txt='/raid/chenby/person_rID/train/unlabel.txt',
                       root_path='/raid/chenby/person_rID/train/images'):
    images = os.listdir(root_path)
    with open(label_txt, "r") as f:
        labeled_images = f.readlines()
    labels = {}
    for line in labeled_images:
        name, id = line.split(':')
        labels[name] = id
    print(len(images), len(labeled_images))
    unlabeled_images = []
    for image in images:
        if image not in labels:
            unlabeled_images.append(image)
    print(len(images), len(labeled_images), len(unlabeled_images))
    with open(unlabel_txt, "w") as f:
        for line in unlabeled_images:
            f.write(line + '\n')

    with open(unlabel_txt, "r") as f:
        unlabeled_images = f.readlines()
        print(len(unlabeled_images))

def select_validation_data(root_path='/raid/chenby/person_rID/2019/REID2019_A',
                           train_txt='/raid/chenby/person_rID/2019/REID2019_A/train_list.txt'):
    print(len(os.listdir(root_path + '/train')))
    with open(train_txt, "r") as f:
        lines = f.readlines()
    labels_dict = defaultdict(list)
    for line in lines:
        img_name, img_label = [i for i in line.replace('\n', '').split()]
        labels_dict[img_label].append(img_name.replace('train/', ''))

    query = []
    gallery = []
    for pid, img_name in labels_dict.items():
        if len(img_name) < 5:
            gallery += img_name
        else:
            query += img_name[:1]
            gallery += img_name[1:]

    print(len(query), len(gallery))

    file = open(os.path.join(root_path,  'query.txt'), 'w')

    for filename in sorted(query):
        file.write(str(filename + '\n'))
    file.close()
    file = open(os.path.join(root_path, 'gallery.txt'), 'w')
    for filename in sorted(gallery):
        file.write(str(filename + '\n'))
    file.close()

if __name__ == '__main__':
    parser = ArgumentParser(description='vis txt result Tool')
    parser.add_argument('--data_dir_query', help='dir to the query datasets')
    parser.add_argument('--data_dir_gallery', help='dir to the gallery datasets')
    parser.add_argument('--save_dir', help='dir to save the generated txt file')
    args = parser.parse_args()
    compute(args.data_dir_query, args.save_dir)
    compute(args.data_dir_gallery, args.save_dir)

    # get_unlabeled_data()
    # select_validation_data()

