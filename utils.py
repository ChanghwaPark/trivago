import pickle as pickle


def save_file(obj, name):
    with open('files/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_file(name):
    with open('files/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
