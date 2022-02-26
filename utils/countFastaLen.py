len_less_200 = 0
len_less_400 = 0
len_less_600 = 0
len_less_800 = 0
len_less_1024 = 0
with open("../data/training_Pid_Pseq_label_screened.txt") as fasta:
    line = fasta.readline()
    while line:
        if(line[0] == '>'):
            uniprot_id = line[1:].strip()
            seq = fasta.readline().strip()
            label = fasta.readline().strip()
            if(len(seq) <= 1024):
                len_less_1024 += 1
            if(len(seq) <= 200):
                len_less_200 += 1
            if(len(seq) <= 400):
                len_less_400 += 1
            if(len(seq) <= 600):
                len_less_600 += 1
            if(len(seq) <= 800):
                len_less_800 += 1
            line = fasta.readline()

print(len_less_200 / 9982) # 0.28
print(len_less_400 / 9982) # 0.65
print(len_less_600 / 9982) # 0.82
print(len_less_800 / 9982) # 0.89
print(len_less_1024 / 9982) # 0.94