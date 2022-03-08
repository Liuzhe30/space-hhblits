import os
import subprocess
import multiprocessing
import time
import logging

blast_path = "/data/psiblast/ncbi-blast-2.12.0+"
psiblast_path = blast_path + "/bin/psiblast"
database_path = blast_path + "/db/uniref50"

working_dir = os.path.abspath('..')
fasta_dir = working_dir + "/data/pssm"
psiblast_out_dir = working_dir + '/data/pssm_res/out'
pssm_dir = working_dir + '/data/pssm_res/ascii_pssm_out'
log_path = working_dir + '/pssm/log'

def get_logger(verbosity=1, name=None):
    # log文件名称(按时间命名)
    filename = "{}.log".format(time.strftime("%Y-%m-%d-%H-%M"))    
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(os.path.join(log_path, filename), "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

logger = get_logger()


"""
bin/psiblast -query /home/zhenxy/space-hhblits/data/pssm/A0FGR8.fa -db db/swissprot -num_iterations 3 -evalue 0.001 -num_threads 2 -out A0FGRB -out_ascii_pssm temp/A0FGRB.pssm
"""

def get_pssm(uniprot_id):
    input_f = os.path.join(fasta_dir, uniprot_id + ".fa")
    output_f = os.path.join(psiblast_out_dir, uniprot_id)
    pssm_output_f = os.path.join(pssm_dir, uniprot_id + ".pssm")
    cmd = "{} -query {} -db {} -num_iterations 3 -evalue 0.001 -num_threads 2 -out {} -out_ascii_pssm {}".format(psiblast_path, input_f, database_path, output_f, pssm_output_f)
    # cmd = "echo {}".format(cmd)
    res = subprocess.call(cmd, shell=True)
    if res == 0:
        logger.info("{} success".format(uniprot_id))
    else:
        logger.info("{} failed".format(uniprot_id))

if __name__ == "__main__":
    
    pool = multiprocessing.Pool(processes=5)
    for _, _ , fasta_files in os.walk(fasta_dir):
        for fasta_f in fasta_files:
            pool.apply_async(get_pssm, (fasta_f.split('.')[0],))

    logger.info("starting")
    pool.close()
    pool.join()
    logger.info("finished")