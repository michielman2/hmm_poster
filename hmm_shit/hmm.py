def train(folder):
    startlist = []
    sequence = []
    filename = 'data/cancer/AF008216.1.fasta'
    with open(filename, "r") as file:
        for line in file:
            sequence += line
