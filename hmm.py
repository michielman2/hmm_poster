import os
import numpy as np
from hmmlearn import hmm

def read_files(cancer_folder, healthy_folder):
   cancer_sequences = []
   healty_sequences = []
    
   for filename in os.listdir(cancer_folder):
      if filename.endswith(".fasta"):
         file_path = os.path.join(cancer_folder, filename)
            
         with open(file_path, 'r') as file:
            sequence = ''
            for line in file:
               line = line.strip()
               if not line.startswith(">"): 
                  sequence += line
            
            cancer_sequences.append(sequence)

   for filename in os.listdir(healthy_folder):
      if filename.endswith(".fasta"):
         file_path = os.path.join(healthy_folder, filename)
            
         with open(file_path, 'r') as file:
            sequence = ''
            for line in file:
               line = line.strip()
               if not line.startswith(">"): 
                  sequence += line
            
            healty_sequences.append(sequence)
    
   return cancer_sequences, healty_sequences

cancer_sequences, healthy_sequences = read_files('data\cancer', 'data\healthy')

# print(cancer_sequences, healthy_sequences)



def encode_sequences(sequences):
   mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
   encoded_sequences = []
   lengths = []
   for seq in sequences:
      encoded = [mapping[nuc] for nuc in seq if nuc in mapping]
      encoded_sequences.extend(encoded)
      lengths.append(len(encoded))

   return np.array(encoded_sequences).reshape(-1, 1), lengths

cancer = encode_sequences(cancer_sequences)
healthy = encode_sequences(healthy_sequences)

# print(cancer)
# print(healthy)

def train_hmm(sequences, n_components=4, n_iter=1000000000):
   encoded_sequences, lengths = encode_sequences(sequences)
   model = hmm.MultinomialHMM(n_components=n_components, n_iter=n_iter, random_state=42)
   model.fit(encoded_sequences, lengths)
   return model

cancer_model = train_hmm(cancer_sequences)
healthy_model = train_hmm(healthy_sequences)

def evaluate_model(model, sequences):
   encoded_sequences, lengths = encode_sequences(sequences)
   log_likelihood = model.score(encoded_sequences, lengths)
   return log_likelihood

cancer_score = evaluate_model(cancer_model, cancer_sequences)
healthy_score = evaluate_model(healthy_model, healthy_sequences)

print(f"cancer Model Score: {cancer_score}")
print(f"healthy Model Score: {healthy_score}")

cancer_score = evaluate_model(cancer_model, healthy_sequences)
healthy_score = evaluate_model(healthy_model, cancer_sequences)

print(f"cancer Model Score: {cancer_score}")
print(f"healthy Model Score: {healthy_score}")

def determine_best(sequence, cancer_model, healthy_model):
   encoded_sequence, lengths = encode_sequences([sequence])
   
   cancer_score = cancer_model.score(encoded_sequence, lengths)
   healthy_score = healthy_model.score(encoded_sequence, lengths)
   
   if cancer_score > healthy_score:
      result = "cancer"
      highest_score = cancer_score
   else:
      result = "healthy"
      highest_score = healthy_score
   
   return result, highest_score

# Example usage
test_sequence = "GGTCTCCCCCAAACCTGCCACCGACGGCCACTTCCGTTTCCCCGATAGTATTTGGGGATCTCGAAGCGATACTTCCGGCTCCCCCCAGGTCCCCAAGCTTTACTTTTGTGGGGCACGACGAGAAAGTCCGCAGCCCCAAACATTCCCAGAGGTCCACTTGGGCCAGTGGTACTTTATCGCAGGGGCAGCTCCCACCAAGGAGGAGTTGGCAACTTTTGACCCTGTGGACAACATTGTCTTCAATATGGCTGCTGGCTCTGCCCCGATGCAGCTCCACCTTCGTGCTACCATCCGCATGAAAGATGGGCTCTGTGTGCCCCGGAAATGGATCTACCACCTGACTGAAGGGAGCACAGATCTCAGAACTGAAGGCCGCCCTGACATGAAGACTGAGCTCTTTTCCAGCTCATGCCCAGGTGGAATCATGCTGAATGAGACAGGCCAGGGTTACCAGCGCTTTCTCCTCTACAATCGCTCACCACATCCTCCCGAAAAGTGTGTGGAGGAATTCAAGTCCCTGACTTCCTGCCTGGACTCCAAAGCCTTCTTATTGACTCCTAGGAATCAAGAGGCCTGTGAGCTGTCCAATAACTGACCTGTAACTTCATCTAAGTCCCCAGATGGGTACAATGGGAGCTGAGTTGTTGGAGGGAGAAGCTGGAGACTTCCAGCTCCAGCTCCCACTCAAGATAATAAAGATAATTTTTCAATCCTCAAAAAAA"
result, score = determine_best(test_sequence, cancer_model, healthy_model)

print("result:" + result)
print("Score: ", score)
