import os
import numpy as np
import math
svd = 1

with open('P17081.hhm') as hhm_file:
    hhm_line = hhm_file.readline()
    while hhm_line:
        if(hhm_line[0:4] == 'LENG'):
            hhm_seq_len = int(hhm_line.split()[1])
            break
        hhm_line = hhm_file.readline()

# befure mutation
with open('P17081.pdb') as pdb_file:
    pdb_matrix = np.zeros([hhm_seq_len, 3], float)
    pdb_line = pdb_file.readline()
    while(pdb_line[0:4] != 'ATOM'):
        pdb_line = pdb_file.readline()
    iddx = 0
    while pdb_line:
        if(pdb_line[0:4] != 'ATOM'):
            break    
        number = pdb_line[22:27].strip()
        CA = pdb_line[13:15]
        if(int(number) == iddx + 1 and CA == 'CA'):
            pdb_matrix[iddx,0] = float(pdb_line[31:38].strip()) # x cood
            pdb_matrix[iddx,1] = float(pdb_line[39:46].strip()) # y cood
            pdb_matrix[iddx,2] = float(pdb_line[47:54].strip()) # z cood
            iddx += 1
        pdb_line = pdb_file.readline()
    #print(pdb_matrix.shape) # seq_len*3
    #print(pdb_matrix)
    
    # calculate distance map
    distance_matrix = np.zeros([hhm_seq_len, hhm_seq_len], float)
    for i in range(hhm_seq_len):
        for j in range(hhm_seq_len):
            distance_matrix[i][j] = math.sqrt(math.pow((pdb_matrix[i][0] - pdb_matrix[j][0]),2) + 
                                                  math.pow((pdb_matrix[i][1] - pdb_matrix[j][1]),2) + 
                                                  math.pow((pdb_matrix[i][2] - pdb_matrix[j][2]),2))
    #print(distance_matrix.shape)
    #print(distance_matrix)
    
    # calculate SVD matrix
    U, s, V = np.linalg.svd(distance_matrix)
    #print(U)
    new_U = U[0:svd]
    svd_matrix = np.dot(new_U, distance_matrix)
    #print(svd_matrix.shape)
    #print(svd_matrix)
    before_matrix = svd_matrix
    
# befure mutation
with open('P17081_Q97R.pdb') as pdb_file:
    pdb_matrix = np.zeros([hhm_seq_len, 3], float)
    pdb_line = pdb_file.readline()
    while(pdb_line[0:4] != 'ATOM'):
        pdb_line = pdb_file.readline()
    iddx = 0
    while pdb_line:
        if(pdb_line[0:4] != 'ATOM'):
            break    
        number = pdb_line[22:27].strip()
        CA = pdb_line[13:15]
        if(int(number) == iddx + 1 and CA == 'CA'):
            pdb_matrix[iddx,0] = float(pdb_line[31:38].strip()) # x cood
            pdb_matrix[iddx,1] = float(pdb_line[39:46].strip()) # y cood
            pdb_matrix[iddx,2] = float(pdb_line[47:54].strip()) # z cood
            iddx += 1
        pdb_line = pdb_file.readline()
    #print(pdb_matrix.shape) # seq_len*3
    #print(pdb_matrix)
    
    # calculate distance map
    distance_matrix = np.zeros([hhm_seq_len, hhm_seq_len], float)
    for i in range(hhm_seq_len):
        for j in range(hhm_seq_len):
            distance_matrix[i][j] = math.sqrt(math.pow((pdb_matrix[i][0] - pdb_matrix[j][0]),2) + 
                                                  math.pow((pdb_matrix[i][1] - pdb_matrix[j][1]),2) + 
                                                  math.pow((pdb_matrix[i][2] - pdb_matrix[j][2]),2))
    #print(distance_matrix.shape)
    #print(distance_matrix)
    
    # calculate SVD matrix
    U, s, V = np.linalg.svd(distance_matrix)
    #print(U)
    new_U = U[0:svd]
    svd_matrix = np.dot(new_U, distance_matrix)
    #print(svd_matrix.shape)
    #print(svd_matrix)
    after_matrix = svd_matrix
    
#print(before_matrix)
#print(after_matrix)
#print(before_matrix.shape)

#print(before_matrix[0][96])
#print(after_matrix[0][96])
    
new_before_matrix = np.zeros([before_matrix.shape[0], before_matrix.shape[1]], float)
new_after_matrix = np.zeros([before_matrix.shape[0], before_matrix.shape[1]], float)
for i in range(before_matrix.shape[1]):
    new_before_matrix[0][i] = before_matrix[0][i] - before_matrix[0][96]
for i in range(after_matrix.shape[1]):
    new_after_matrix[0][i] = after_matrix[0][i] - after_matrix[0][96]
    
#print(new_before_matrix)
#print(new_after_matrix)

delta_array = []
for i in range(96-10,96+10+1):
    delta_array.append(new_after_matrix[0][i] - new_before_matrix[0][i])
#print(delta_array)

for i in range(0, 21):
    if(i == 10):
        continue
    if(math.fabs(delta_array[i]) <= 1):
        print('select /P17081//A/' + str(87+i) +';')
        print("color blue, sele;")
    elif(math.fabs(delta_array[i]) <= 2):
        print('select /P17081//A/' + str(87+i) +';')
        print("color green, sele;")
    else:
        print('select /P17081//A/' + str(87+i) +';')
        print("color yellow, sele;")        