# one-hot dict for proteins
import os
import numpy as np
eyes = np.eye(20)
protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
        'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
        'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
        'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}
#print(protein_dict)
# generate sliding window dataset, size = 61 (30+1+30)
window_length = 61
shhm_path = 'data/shhm_data/0A_simple_average_shhm_exp/'
all_sample_x = []
all_sample_y = []
t = int((window_length - 1) / 2)
with open("data/dataset_alphafold.txt") as file:
    line = file.readline()
    while line:
        label_list = []
        if(line[0] == '>'):
            uniprot_id = line[1:].strip()
            seq = file.readline().strip()
            label = file.readline().strip()
            seq_len = len(seq)
            feature_matrix = np.zeros([seq_len + 2 * t, 52], float) # 20+30+1+1,onehot+spacehhblits+noseq+mask
            shhm_matrix = np.loadtxt(shhm_path + uniprot_id + ".shhm")
            # check padding
            for i in range(feature_matrix.shape[0]):
                if(i < t):
                    feature_matrix[i][-2] = 1 #noseq
                elif(i >= seq_len + t):
                    feature_matrix[i][-2] = 1 #noseq
                else:
                    feature_matrix[i,0:20] = protein_dict[seq[i - t]]
                    feature_matrix[i,20:50] = shhm_matrix[i - t,:]
                    feature_matrix[i,-1] = 1 # mask
        #print(feature_matrix.shape) # seq_len + 2 * t, 52
        # sliding window
        top = 0
        buttom = window_length
        while(buttom <= feature_matrix.shape[0]):
            all_sample_x.append(feature_matrix[top:buttom])            
            top += 1
            buttom += 1
        for i in range(seq_len):
            all_sample_y.append(int(label[i]))
        #print(np.array(sample_list).shape)
        #print(np.array(label_list).shape)
        line = file.readline()        
all_sample_x = np.array(all_sample_x)
all_sample_y = np.array(all_sample_y)
print(all_sample_x.shape)
print(all_sample_y.shape)
np.save("data/dataset_0A_simple_average_shhm_exp_slidingwindow_x.npy", all_sample_x)
np.save("data/dataset_0A_simple_average_shhm_exp_slidingwindow_y.npy", all_sample_y)