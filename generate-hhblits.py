# fetch name list
import os
hhm_path = 'data/hhblits_example/'
pdb_path = 'data/pdb_example/'
hhm_path_files = os.listdir(hhm_path)  
name_list = []
for fi in hhm_path_files: 
    hhm_name = fi.split('.')[0]
    name_list.append(hhm_name)
print(len(name_list))

# generate mean space-hhblits
import numpy as np
import math
max_range = 0
output_path = 'data/shhm_data/0A_simple_average_shhm_exp/'
for uniprot_id in name_list:
    # fetch length
    with open(hhm_path + uniprot_id + '.hhm') as hhm_file:
        hhm_line = hhm_file.readline()
        while hhm_line:
            if(hhm_line[0:4] == 'LENG'):
                hhm_seq_len = int(hhm_line.split()[1])
                break
            hhm_line = hhm_file.readline()
    # fetch 30d feature from .hhm    
    with open(hhm_path + uniprot_id + '.hhm') as hhm_file:     
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
                        #hhm_matrix[idxx - 1, j] = int(each_item[j])
                        hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j])/2000))                      
                    except IndexError:
                        pass
            elif(len(hhm_line.split()) == 10):
                each_item = hhm_line.split()[0:10]
                for idx, s in enumerate(each_item):
                    if(s == '*'):
                        each_item[idx] = '99999'                             
                for j in range(20, 30):
                    try:
                        #hhm_matrix[idxx - 1, j] = int(each_item[j - 20]) 
                        hhm_matrix[idxx - 1, j] = 10/(1 + math.exp(-1 * int(each_item[j - 20])/2000))                       
                    except IndexError:
                        pass                            
            hhm_line = hhm_file.readline()
        #print(hhm_matrix.shape) # # seq_len*30
        #print(hhm_matrix)
    
    # fetch Ca 3d-coord from .pdb
    with open(pdb_path + uniprot_id + '.pdb') as pdb_file:
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
        
    # spatial filtering
    space_hhm_matrix = np.zeros([hhm_seq_len, 30], float)
    res_dict = {}
    for residue_num in range(hhm_seq_len):
        res_dict[residue_num] = []
        x, y, z = pdb_matrix[residue_num,0],pdb_matrix[residue_num,1],pdb_matrix[residue_num,2]
        for pair in range(hhm_seq_len):
            x_pair, y_pair, z_pair = pdb_matrix[pair,0],pdb_matrix[pair,1],pdb_matrix[pair,2]
            if((x-x_pair)*(x-x_pair) + (y-y_pair)*(y-y_pair) + (z-z_pair)*(z-z_pair) <= max_range*max_range):
                res_dict[residue_num].append(pair)
    for residue_num in range(hhm_seq_len):
        residue_list = res_dict[residue_num]
        for j in range(30):
            for num in residue_list:
                space_hhm_matrix[residue_num][j] += hhm_matrix[num][j]
                space_hhm_matrix[residue_num][j] /= float(len(residue_list)) # average
    print(space_hhm_matrix.shape)
    print(space_hhm_matrix)
    
    with open(output_path + uniprot_id + '.shhm','w+') as out_file:
        np.savetxt(out_file, space_hhm_matrix, fmt='%.6f')
            