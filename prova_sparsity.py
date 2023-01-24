from ltnrec.data import DataManager
import pickle

# create dataset manager
data_manager = DataManager("./datasets")
# ml = data_manager.get_ml100k_folds(0, mode="ml\mr")
# print(len(ml.train))
# ml = data_manager.increase_data_sparsity(ml, 0.05, 0)
# print(len(ml.train))

with open("./sparsity-exp-mr/datasets/mr-mr(movies)-100-0.10-seed_0", 'rb') as dataset_file:
    dataset = pickle.load(dataset_file)

print(dataset.val[56][23])
print(len(dataset.train))

with open("./sparsity-exp-mr/datasets/mr-mr(movies+genres)-100-0.10-seed_0", 'rb') as dataset_file:
    dataset = pickle.load(dataset_file)

print(dataset.val[56][23])
print(len(dataset.train))
