import os
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

# (|new-old|)/old
hhm_seq_len = 205
with open('P17081.hhm') as hhm_file:     
    hhm_matrix = np.zeros([hhm_seq_len, 30], float)
    hhm_line = hhm_file.readline()
    idxx = 0
    while(hhm_line[0] != '#'):
        hhm_line = hhm_file.readline()
    for i in range(0,5):
        hhm_line = hhm_file.readline()
    while hhm_line:
        if(len(hhm_line.split()) == 23):
            idxx += 1
            if(idxx == hhm_seq_len + 1):
                break
            each_item = hhm_line.split()[2:22]
            for idx, s in enumerate(each_item):
                if(s == '*'):
                    each_item[idx] = '99999'                            
            for j in range(0, 20):
                try:
                    hhm_matrix[idxx - 1, j] = int(each_item[j])
                    #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))                                              
                except IndexError:
                    pass
        elif(len(hhm_line.split()) == 10):
            each_item = hhm_line.split()[0:10]
            for idx, s in enumerate(each_item):
                if(s == '*'):
                    each_item[idx] = '99999'                             
            for j in range(20, 30):
                try:
                    hhm_matrix[idxx - 1, j] = int(each_item[j - 20]) 
                    #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j - 20])/2000))                                               
                except IndexError:
                    pass                            
        hhm_line = hhm_file.readline()
origin_hhblits = hhm_matrix[96-15:96+16]
print(origin_hhblits)

with open('P17081_Q97R.hhm') as hhm_file:     
    hhm_matrix = np.zeros([hhm_seq_len, 30], float)
    hhm_line = hhm_file.readline()
    idxx = 0
    while(hhm_line[0] != '#'):
        hhm_line = hhm_file.readline()
    for i in range(0,5):
        hhm_line = hhm_file.readline()
    while hhm_line:
        if(len(hhm_line.split()) == 23):
            idxx += 1
            if(idxx == hhm_seq_len + 1):
                break
            each_item = hhm_line.split()[2:22]
            for idx, s in enumerate(each_item):
                if(s == '*'):
                    each_item[idx] = '99999'                            
            for j in range(0, 20):
                try:
                    hhm_matrix[idxx - 1, j] = int(each_item[j])
                    #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))                                              
                except IndexError:
                    pass
        elif(len(hhm_line.split()) == 10):
            each_item = hhm_line.split()[0:10]
            for idx, s in enumerate(each_item):
                if(s == '*'):
                    each_item[idx] = '99999'                             
            for j in range(20, 30):
                try:
                    hhm_matrix[idxx - 1, j] = int(each_item[j - 20]) 
                    #hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j - 20])/2000))                                               
                except IndexError:
                    pass                            
        hhm_line = hhm_file.readline()
mut_hhblits = hhm_matrix[96-15:96+16]
#print(mut_hhblits)

de_hhblits = (mut_hhblits - origin_hhblits) / origin_hhblits
#deabs_hhblits = np.absolute(de_hhblits)
#print(deabs_hhblits)

spa_ori_hhblits = np.loadtxt("P17081.shhm")
spa_mut_hhblits = np.loadtxt("P17081_Q97R.shhm")
de_spa_hhblits = (spa_mut_hhblits[96-15:96+16] - spa_ori_hhblits[96-15:96+16]) / spa_ori_hhblits[96-15:96+16]
#deabs_spa_hhblits = np.absolute(de_spa_hhblits)
de_spa_hhblits[de_spa_hhblits > 0.4] = 0.4
#print(de_spa_hhblits)

# fetch Ca 3d-coord from .pdb
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
ori_coord = pdb_matrix[96-15:96+16]

# fetch Ca 3d-coord from .pdb
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
mut_coord = pdb_matrix[96-15:96+16]
de_coord = (mut_coord - ori_coord) / ori_coord

distance_mut = np.zeros([31,1],float)
for i in range(31):
    distance_mut[i][0] = math.pow(mut_coord[i][0] - mut_coord[15][0],2)+math.pow(mut_coord[i][1] - mut_coord[15][1],2)+math.pow(mut_coord[i][2] - mut_coord[15][2],2)
distance_ori = np.zeros([31,1],float)
for i in range(31):
    distance_ori[i][0] = math.pow(ori_coord[i][0] - ori_coord[15][0],2)+math.pow(ori_coord[i][1] - ori_coord[15][1],2)+math.pow(ori_coord[i][2] - ori_coord[15][2],2)
del_distance = distance_mut - distance_ori
del_distance[30][0] = 1.8
print(del_distance)

#sns.heatmap(de_hhblits,cmap="coolwarm")
#sns.heatmap(de_spa_hhblits,cmap="coolwarm")
#sns.heatmap(de_coord,cmap="coolwarm")
sns.heatmap(del_distance,cmap="coolwarm")
plt.show()

