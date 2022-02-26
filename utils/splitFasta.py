with open("../data/training_Pid_Pseq_label_screened.txt") as fasta:
    line = fasta.readline()
    while line:
        if(line[0] == '>'):
            uniprot_id = line[1:].strip()
            seq = fasta.readline().strip()
            label = fasta.readline().strip()
            new = open("../data/fasta_all/" + uniprot_id + ".fasta", "w+")
            new.write(">" + uniprot_id + '\n')
            new.write(seq)
            line = fasta.readline()