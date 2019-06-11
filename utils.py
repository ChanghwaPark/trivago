import os
import pickle as pickle
import shutil


def save_file(obj, name):
    with open('files/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_file(name):
    with open('files/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_model(saver, M, model_dir, global_step):
    path = saver.save(M.sess, os.path.join(model_dir, 'model'), global_step=global_step)
    print(f"Saving model to {path}")


def delete_existing(path):
    if os.path.exists(path):
        shutil.rmtree(path)
