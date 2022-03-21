# one-hot dict for proteins
import os
import numpy as np
import multiprocessing
def generate_data(distance,filter_strategy,partition):
    """generate npy data for generator

    Args:
        distance (int): 0 1 3 5 7 9 11
        filter_strategy (str): 'simple_average'
                                'weighted_average'
                                'weighted_average_square'
                                'weighted_average_exponent'
                                'simple_average_frequency'
        partition (str): 'training' 
                         'validation'
                         'test'
    """
    eyes = np.eye(20)
    protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
            'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
            'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
            'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}
    #print(protein_dict)
    # generate sliding window dataset, size = 61 (30+1+30)
    window_length = 31
    # shhm_path = '/home/panwh/space-hhblits/data/distance_2555_data'
    shhm_path = 'data/shhm_data/' + str(distance) + 'A_' + filter_strategy + '_shhm/'
    # shhm_path = 'data/shhm_data/0A_simple_average_shhm/'
    all_sample_x = []
    all_sample_y = []
    t = int((window_length - 1) / 2)
    file_path = 'data/dataset_alphafold_' + partition + '.txt'
    with open(file_path) as file:
        line = file.readline()
        while line:
            label_list = []
            if(line[0] == '>'):
                uniprot_id = line[1:].strip()
                seq = file.readline().strip()
                label = file.readline().strip()
                seq_len = len(seq)
                feature_matrix = np.zeros([seq_len + 2 * t, 22+30], float) # 20+(dis:2555/1/5/10/20)+1+1,onehot+distancemap+noseq+mask
                shhm_matrix = np.loadtxt(os.path.join(shhm_path, uniprot_id + ".shhm"))
                # check padding
                for i in range(feature_matrix.shape[0]):
                    if(i < t):
                        feature_matrix[i][-2] = 1 #noseq
                    elif(i >= seq_len + t):
                        feature_matrix[i][-2] = 1 #noseq
                    else:
                        feature_matrix[i,0:20] = protein_dict[seq[i - t]]
                        feature_matrix[i,20:20+30] = shhm_matrix[i - t,:]
                        feature_matrix[i,-1] = 1 # mask
            print(feature_matrix.shape) # seq_len + 2 * t, 22+2555
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
    # print(all_sample_x.shape)
    # print(all_sample_y.shape)
    output_path_x = '/data/pwh/dataset_' + str(distance) + 'A_' + filter_strategy + '_slidingwindow_' + partition + '_x.npy'
    output_path_y = '/data/pwh/dataset_' + str(distance) + 'A_' + filter_strategy + '_slidingwindow_' + partition + '_y.npy'
    # output_path_x = 'data/dataset_' + 'max_length_distance_' + partition + '_slidingwindow_x.npy'
    # output_path_y = 'data/dataset_' + 'max_length_distance_' + partition + '_slidingwindow_y.npy'
    np.save(output_path_x, all_sample_x)
    np.save(output_path_y, all_sample_y)
    # print(distance, filter_strategy, partition)


def multi_generate_data_wrapper(args):
    return generate_data(*args)

if __name__ == '__main__':
    # a = (0, 'simple_average', 'training')
    # generate_data(*a)
    multiprocessing_list = []
    # multiprocessing_list.append(('training'))
    # multiprocessing_list.append(('validation'))
    generate_data(0, 'simple_average', 'training')
    generate_data(1, 'simple_average', 'training')
    generate_data(3, 'simple_average', 'training')
    generate_data(5, 'simple_average', 'training')
    generate_data(7, 'simple_average', 'training')
    generate_data(9, 'simple_average', 'training')
    generate_data(11, 'simple_average', 'training')
    # multiprocessing_list.append((1, 'weighted_average', 'training'))
    # multiprocessing_list.append((3, 'weighted_average', 'training'))
    # multiprocessing_list.append((5, 'weighted_average', 'training'))
    # multiprocessing_list.append((7, 'weighted_average', 'training'))
    # multiprocessing_list.append((9, 'weighted_average', 'training'))
    # multiprocessing_list.append((11, 'weighted_average', 'training'))
    # multiprocessing_list.append((1, 'weighted_average_square', 'training'))
    # multiprocessing_list.append((3, 'weighted_average_square', 'training'))
    # multiprocessing_list.append((5, 'weighted_average_square', 'training'))
    # multiprocessing_list.append((7, 'weighted_average_square', 'training'))
    # multiprocessing_list.append((9, 'weighted_average_square', 'training'))
    # multiprocessing_list.append((11, 'weighted_average_square', 'training'))
    # multiprocessing_list.append((1, 'weighted_average_exponent', 'training'))
    # multiprocessing_list.append((3, 'weighted_average_exponent', 'training'))
    # multiprocessing_list.append((5, 'weighted_average_exponent', 'training'))
    # multiprocessing_list.append((7, 'weighted_average_exponent', 'training'))
    # multiprocessing_list.append((9, 'weighted_average_exponent', 'training'))
    # multiprocessing_list.append((11, 'weighted_average_exponent', 'training'))

    generate_data(0, 'simple_average', 'validation')
    generate_data(1, 'simple_average', 'validation')
    generate_data(3, 'simple_average', 'validation')
    generate_data(5, 'simple_average', 'validation')
    generate_data(7, 'simple_average', 'validation')
    generate_data(9, 'simple_average', 'validation')
    generate_data(11, 'simple_average', 'validation')
    # multiprocessing_list.append((1, 'weighted_average', 'validation'))
    # multiprocessing_list.append((3, 'weighted_average', 'validation'))
    # multiprocessing_list.append((5, 'weighted_average', 'validation'))
    # multiprocessing_list.append((7, 'weighted_average', 'validation'))
    # multiprocessing_list.append((9, 'weighted_average', 'validation'))
    # multiprocessing_list.append((11, 'weighted_average', 'validation'))
    # multiprocessing_list.append((1, 'weighted_average_square', 'validation'))
    # multiprocessing_list.append((3, 'weighted_average_square', 'validation'))
    # multiprocessing_list.append((5, 'weighted_average_square', 'validation'))
    # multiprocessing_list.append((7, 'weighted_average_square', 'validation'))
    # multiprocessing_list.append((9, 'weighted_average_square', 'validation'))
    # multiprocessing_list.append((11, 'weighted_average_square', 'validation'))
    # multiprocessing_list.append((1, 'weighted_average_exponent', 'validation'))
    # multiprocessing_list.append((3, 'weighted_average_exponent', 'validation'))
    # multiprocessing_list.append((5, 'weighted_average_exponent', 'validation'))
    # multiprocessing_list.append((7, 'weighted_average_exponent', 'validation'))
    # multiprocessing_list.append((9, 'weighted_average_exponent', 'validation'))
    # multiprocessing_list.append((11, 'weighted_average_exponent', 'validation'))

    generate_data(0, 'simple_average', 'test')
    generate_data(1, 'simple_average', 'test')
    generate_data(3, 'simple_average', 'test')
    generate_data(5, 'simple_average', 'test')
    generate_data(7, 'simple_average', 'test')
    generate_data(9, 'simple_average', 'test')
    generate_data(11, 'simple_average', 'test')
    # multiprocessing_list.append((1, 'weighted_average', 'test'))
    # multiprocessing_list.append((3, 'weighted_average', 'test'))
    # multiprocessing_list.append((5, 'weighted_average', 'test'))
    # multiprocessing_list.append((7, 'weighted_average', 'test'))
    # multiprocessing_list.append((9, 'weighted_average', 'test'))
    # multiprocessing_list.append((11, 'weighted_average', 'test'))
    # multiprocessing_list.append((1, 'weighted_average_square', 'test'))
    # multiprocessing_list.append((3, 'weighted_average_square', 'test'))
    # multiprocessing_list.append((5, 'weighted_average_square', 'test'))
    # multiprocessing_list.append((7, 'weighted_average_square', 'test'))
    # multiprocessing_list.append((9, 'weighted_average_square', 'test'))
    # multiprocessing_list.append((11, 'weighted_average_square', 'test'))
    # multiprocessing_list.append((1, 'weighted_average_exponent', 'test'))
    # multiprocessing_list.append((3, 'weighted_average_exponent', 'test'))
    # multiprocessing_list.append((5, 'weighted_average_exponent', 'test'))
    # multiprocessing_list.append((7, 'weighted_average_exponent', 'test'))
    # multiprocessing_list.append((9, 'weighted_average_exponent', 'test'))
    # multiprocessing_list.append((11, 'weighted_average_exponent', 'test'))

    # pool_obj = multiprocessing.Pool()
    # pool_obj.map(multi_generate_data_wrapper, [idx for idx in multiprocessing_list])