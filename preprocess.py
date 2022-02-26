def data_screen(input_file_path, output_file_path, aa_set, aa_length):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    output_list = []
    for idx in range(0, len(lines), 3):
        aa_sequence = lines[idx + 1].strip()
        if len(aa_sequence) >= aa_length and set(aa_sequence).issubset(aa_set):
            output_list.append(lines[idx])
            output_list.append(lines[idx+1])
            output_list.append(lines[idx+2])
        else:
            pass

    with open(output_file_path,'w') as output_file:
        for line in output_list:
            output_file.write(line)

if __name__ == '__main__':
    input_file_path = r'D:\ppp\hhblits_space\data\training_Pid_Pseq_label.txt'
    output_file_path = r'D:\ppp\hhblits_space\data\training_Pid_Pseq_label_screened.txt'
    aa_set = set('ARNDCQEGHILKMFPSTWYV')
    aa_length = 30
    data_screen(input_file_path, output_file_path, aa_set, aa_length)
