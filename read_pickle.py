import pickle

with open('Data/Train.pickle','rb') as file:
    data = pickle.load(file)

print(data)
