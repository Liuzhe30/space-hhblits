# fetch name list
import os
hhm_path = 'data/hhblits_example/'
pdb_path = 'data/pdb_example/'
dssp_path = 'data/dssp/'
hhm_path_files = os.listdir(hhm_path)  
name_list = []
for fi in hhm_path_files: 
    hhm_name = fi.split('.')[0]
    name_list.append(hhm_name)
print(len(name_list))

for i in name_list:
    os.system("mkdssp -i " + pdb_path + i + '.pdb -o ' + dssp_path + i + '.dssp')