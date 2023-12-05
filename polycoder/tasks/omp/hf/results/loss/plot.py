import glob
import os
import matplotlib.pyplot as plt

files = [file for file in os.listdir() if file.endswith('.log')]

def parse_file(file):
    Y = []

    with open(file, 'r') as f:
        for line in f:
            Y.append(float(line[line.rfind(':')+1:]))

    return list(range(len(Y)))[10:], Y[10:]


fig, ax = plt.subplots(figsize=(12, 8))

for file in files:
    X, Y = parse_file(file)
    plt.plot(X, Y, label=file[:-4])

plt.xlabel('time step')
plt.ylabel('loss')

plt.legend()
# plt.show()
plt.savefig('loss_graph.jpeg')
