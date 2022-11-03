# todo ora bisogna implementare l'esperimento con l'aggiunta dei generi
# todo sarebbe curioso simulare il cold-start -> non credo si possa utilizzare diag perche' se no i casi di cold-start
#  non vengono iterati
from ltnrec.data import MovieLensMR
from ltnrec.loaders import ValDataLoader, TrainingDataLoaderLTN
from ltnrec.models import MatrixFactorization, LTNTrainerMF
from torch.optim import Adam

data = MovieLensMR("./datasets")
ml_tr, ml_val, ml_test, ml_n_users, ml_n_items = data.get_ml100k_folds(123)
ratings, _, _, ml_to_new_idx, _, user_mapping_ml, _ = data.create_ml_mr_fusion()
fusion_train, fusion_val, fusion_test, fusion_n_users, fusion_n_items = \
    data.get_fusion_folds(ratings, ml_to_new_idx, user_mapping_ml, ml_val, ml_test)
val_loader_ml = ValDataLoader(ml_val, 512)
val_loader_fusion = ValDataLoader(fusion_val, 512)
test_loader_ml = ValDataLoader(ml_test, 512)
test_loader_fusion = ValDataLoader(fusion_test, 512)
tr_loader_ml = TrainingDataLoaderLTN(ml_tr, 256)
tr_loader_fusion = TrainingDataLoaderLTN(fusion_train, 256)
mf_model_ml = MatrixFactorization(ml_n_users, ml_n_items, 1, biased=True)
mf_model_fusion = MatrixFactorization(fusion_n_users, fusion_n_items, 1, biased=True)
trainer_ml = LTNTrainerMF(mf_model_ml, Adam(mf_model_ml.parameters(), lr=0.001, weight_decay=0.0001), 0.05)
trainer_fusion = LTNTrainerMF(mf_model_fusion, Adam(mf_model_fusion.parameters(), lr=0.001, weight_decay=0.0001), 0.05)
trainer_ml.train(tr_loader_ml, val_loader_ml, "hit@10", n_epochs=50, early=10, verbose=1)
trainer_fusion.train(tr_loader_fusion, val_loader_fusion, "hit@10", n_epochs=50, early=10, verbose=1)
print(trainer_ml.test(test_loader_ml, ["hit@10", "ndcg@10"]))
print(trainer_fusion.test(test_loader_fusion, ["hit@10", "ndcg@10"]))
