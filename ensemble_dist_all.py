import numpy as np
import os
import json


base_dir = './output_B/'
model_paths = ['resnet101_ibn_a_add-2019', 'efn-b2_add-2019_circle', 'efn-b0_add-2019_circle_triplet', 'efn-b1_add-2019_circle',
               'efn-b1_add-2019_circle_triplet']

query = np.load(base_dir + model_paths[0] + '/query_path_all.npy')
gallery = np.load(base_dir + model_paths[0] + '/gallery_path_all.npy')
print(len(query), len(gallery))


for i, model_path in enumerate(model_paths):
    if i == 0:
        distmat = 4 * np.load(base_dir + model_path + '/distmat_all.npy')
    else:
        distmat += np.load(base_dir + model_path + '/distmat_all.npy')

print('distmat:', distmat.shape)
indexes = np.argsort(distmat, axis=1)

res = {}
for idx, index in enumerate(indexes):
    query_t = os.path.basename(query[idx])
    gallery_t = [os.path.basename(i) for i in gallery[index][:200].tolist()]
    res[query_t] = gallery_t

ensamble_path = ''
for i, p in enumerate(model_paths):
    ensamble_path += p + '_'

save_path = 'output_B/ensamble/all_' + ensamble_path + 'submit.json'
print("Writing to {}".format(save_path))
json.dump(res, open(save_path, 'w'))
