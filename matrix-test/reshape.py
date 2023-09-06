import torch

a = torch.tensor([[1,2], [3,4], [5,6]])

# [1 2]
# [3 4]
# [5 6]
# m x n = 3 x 2 matrise
print("--------------------------------")
print("Opprinnelig matrise:")
print(a)
print("--------------------------------")
print("1. reshape(-1):")
print(a.reshape(-1)) 
# Flater ut matrisen slik at den blir 1 x (antall elementer) matrise:Blir: [1, 2, 3, 4, 5, 6] 1 x 6 matrise
print("--------------------------------")

print("2. reshape(-1, 3):")
print(a.reshape(-1, 3)) 
# -1 tilpasses andre tallet, i det tilfelle 
# for å få til m x 3 matrise, må -1 bli til 2!

print("--------------------------------")

# For å transponerematrise m x n til n x m, reshape slik (n, m) eller (-1, m):
# Eksempel:
# Vi har 3 x 2 matrise, så for å transponere bruker vi: 
# reshape(2, 3) eller reshape(-1, 3):
print("3. reshape(2, 3) og reshape(-1, 3):")
print(a.reshape(2, 3)) 
print(a.reshape(-1, 3)) 


