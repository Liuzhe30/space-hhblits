# fetch name list
import os
hhm_path = 'data/hhblits_example/'
pssm_path = 'data/psiblast_output/'
hhm_path_files = os.listdir(hhm_path)  
name_list = []
for fi in hhm_path_files: 
    hhm_name = fi.split('.')[0]
    name_list.append(hhm_name)
print(len(name_list))

import numpy as np
import math
max_range = 0
output_path = 'data/pssm_data/'
for uniprot_id in name_list:
    # fetch length
    #print(uniprot_id)
    with open(hhm_path + uniprot_id + '.hhm') as hhm_file:
        hhm_line = hhm_file.readline()
        while hhm_line:
            if(hhm_line[0:4] == 'LENG'):
                hhm_seq_len = int(hhm_line.split()[1])
                break
            hhm_line = hhm_file.readline()
    
    # fetch pssm from .pssm
    with open(pssm_path + uniprot_id + '.pssm') as pssm:
        pssm_matrix = np.zeros([hhm_seq_len, 20], int)
        filelines = pssm.readlines()
        i = 0
        for pssm_line in filelines:
            if(len(pssm_line.split()) == 44):
                each_item = pssm_line.split()[2:22]
                #print(each_item)
                for j in range(0, 20):
                    try:
                        pssm_matrix[i, j] = int(each_item[j])
                    except IndexError:
                        pass
                i += 1
        #print(pssm_matrix.shape) # seq_len*3
        #print(pssm_matrix)
    
    with open(output_path + uniprot_id + '.npy','w+') as out_file:
        np.savetxt(out_file, pssm_matrix, fmt='%.3f')
