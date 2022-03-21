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
output_path = 'data/coord_data/'
for uniprot_id in name_list:
    # fetch length
    print(uniprot_id)
    with open(hhm_path + uniprot_id + '.hhm') as hhm_file:
        hhm_line = hhm_file.readline()
        while hhm_line:
            if(hhm_line[0:4] == 'LENG'):
                hhm_seq_len = int(hhm_line.split()[1])
                break
            hhm_line = hhm_file.readline()
    
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
    
    with open(output_path + uniprot_id + '.npy','w+') as out_file:
        np.savetxt(out_file, pdb_matrix, fmt='%.4f')
