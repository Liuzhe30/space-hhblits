import os
import random
name_list = []
with open("../data/dataset_alphafold.txt") as dataset_file:
    line = dataset_file.readline()
    while line:
        if(line[0] == '>'):
            uniprot_id = line[1:].strip()
            name_list.append(uniprot_id)
        line = dataset_file.readline()
random.shuffle(name_list)        
with open("../data/dataset_alphafold_shuffle.txt","w+") as shuffle_file:
    for name in name_list:
        with open("../data/dataset_alphafold.txt") as dataset_file:
            line = dataset_file.readline()
            while line:
                if(line[0] == '>'):
                    uniprot_id = line[1:].strip()
                    seq = dataset_file.readline().strip()
                    label = dataset_file.readline().strip()  
                    if(uniprot_id == name):
                        shuffle_file.write(">" + uniprot_id + '\n')
                        shuffle_file.write(seq + '\n')
                        shuffle_file.write(label + '\n')
                line = dataset_file.readline()
                