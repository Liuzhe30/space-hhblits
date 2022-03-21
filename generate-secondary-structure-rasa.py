import os
import numpy as np
import math
maxasa_dict = {
'C':167, 'D':193, 'S':155, 'Q':225, 'K':236,
'I':197, 'P':159, 'T':172, 'F':240, 'N':195,
'G':104, 'H':224, 'L':201, 'R':274, 'W':285,
'A':129, 'V':174, 'E':223, 'Y':263, 'M':224
}
maxasa_dict2 = {
'C':148, 'D':187, 'S':143, 'Q':214, 'K':230,
'I':195, 'P':154, 'T':163, 'F':228, 'N':187,
'G':97, 'H':203, 'L':191, 'R':265, 'W':264,
'A':121, 'V':165, 'E':214, 'Y':255, 'M':203
}
eyes = np.eye(3)
ss_dict = {
'H':eyes[0],'G':eyes[0],'I':eyes[0],
'B':eyes[1],'E':eyes[1],
'T':eyes[2]
}

# fetch name list

hhm_path = 'data/hhblits_example/'
dssp_path = 'data/dssp/'
ss_path = 'data/ss/'
rasa_path = 'data/rasa/'
rasa_path2 = 'data/rasa2/'
asa_path = 'data/asa/'
hhm_path_files = os.listdir(hhm_path)  
name_list = []
for fi in hhm_path_files: 
    hhm_name = fi.split('.')[0]
    name_list.append(hhm_name)
print(len(name_list))
for name in name_list:
    # fetch len
    print(name)
    with open(hhm_path + name + '.hhm') as hhm_file:
        hhm_line = hhm_file.readline()
        while hhm_line:
            if(hhm_line[0:4] == 'LENG'):
                hhm_seq_len = int(hhm_line.split()[1])
                break
            hhm_line = hhm_file.readline()
            
    with open(dssp_path + name + '.dssp') as dssp_file:
        ss_matrix = np.zeros([hhm_seq_len, 3], int)
        rasa_matrix = np.zeros([hhm_seq_len, 1], float)
        rasa_matrix2 = np.zeros([hhm_seq_len, 1], float)
        asa_matrix = np.zeros([hhm_seq_len, 1], float)
        line = dssp_file.readline()
        while line:
            if(line.split()[0] == '#'):
                break
            line = dssp_file.readline()
        #ss_file = open(ss_path + name + '.ss','w+')
        #rasa_file = open(rasa_path + name + '.rasa', 'w+')
        line = dssp_file.readline()
        index = 0
        while line:
            if(len(line.split()) > 0):
                SS = line[16]
                ACC = int(line[35:38].strip())
                AA = line[13]
                # check ss
                if(SS != ' '):
                    if(SS not in ss_dict.keys()):
                        ss_matrix[index][0:3] = eyes[2]
                    else:
                        ss_matrix[index][0:3] = ss_dict[SS]
                # check rasa
                rasa_matrix[index][0] = float(ACC)/maxasa_dict[AA]
                rasa_matrix2[index][0] = float(ACC)/maxasa_dict2[AA]
                asa_matrix[index][0] = float(ACC)
                index += 1                    
            line = dssp_file.readline()
        #print(ss_matrix)
        print(ss_matrix.shape)
        #print(rasa_matrix)
        print(rasa_matrix.shape)
        with open(ss_path + name + '.npy','w+') as out_file:
            np.savetxt(out_file, ss_matrix, fmt='%.1f')
        with open(rasa_path + name + '.npy','w+') as out_file:
            np.savetxt(out_file, rasa_matrix, fmt='%.3f')
        with open(rasa_path2 + name + '.npy','w+') as out_file:
            np.savetxt(out_file, rasa_matrix2, fmt='%.3f')
        with open(asa_path + name + '.npy','w+') as out_file:
            np.savetxt(out_file, asa_matrix, fmt='%.3f')