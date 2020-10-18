import numpy as np
import os
import json


base_dir = './output/'
model_paths = ['efn-b0', 'efn-b2', 'efn-b3', 'efn-b5', 'efn-b6', 'efn-b7']

query_1 = np.load(base_dir + model_paths[0] + '/query_path_1.npy')
gallery_1 = np.load(base_dir + model_paths[0] + '/gallery_path_1.npy')
print(len(query_1), len(gallery_1))
query_2 = np.load(base_dir + model_paths[0] + '/query_path_2.npy')
gallery_2 = np.load(base_dir + model_paths[0] + '/gallery_path_2.npy')
print(len(query_2), len(gallery_2))

for i, model_path in enumerate(model_paths):
    if i == 0:
        distmat_1 = np.load(base_dir + model_path + '/distmat_1.npy')
    else:
        distmat_1 += np.load(base_dir + model_path + '/distmat_1.npy')

print('distmat_1:', distmat_1.shape)
indexes = np.argsort(distmat_1, axis=1)

res_1 = {}
for idx, index in enumerate(indexes):
    query = os.path.basename(query_1[idx])
    gallery = [os.path.basename(i) for i in gallery_1[index][:200].tolist()]
    res_1[query] = gallery

for i, model_path in enumerate(model_paths):
    if i == 0:
        distmat_2 = np.load(base_dir + model_path + '/distmat_2.npy')
    else:
        distmat_2 += np.load(base_dir + model_path + '/distmat_2.npy')

print('distmat_2:', distmat_2.shape)

indexes = np.argsort(distmat_2, axis=1)

res_2 = {}
for idx, index in enumerate(indexes):
    query = os.path.basename(query_2[idx])
    gallery = [os.path.basename(i) for i in gallery_2[index][:200].tolist()]
    res_2[query] = gallery

data = dict()
for k, v in res_1.items():
    data[k] = v
for k, v in res_2.items():
    data[k] = v

ensamble_path = ''
for i, p in enumerate(model_paths):
    ensamble_path += p + '_'

save_path = 'output/ensamble/' + ensamble_path + 'submit.json'
print("Writing to {}".format(save_path))
json.dump(data, open(save_path, 'w'))
