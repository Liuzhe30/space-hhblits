# space-hhblits
Evaluation on different combinations of AlphaFold2 (coordinates) and HHblits.

## Download data
We provide the dataset used in this study,  you can download [dataset_alphafold.txt](https://github.com/Liuzhe30/space-hhblits/blob/main/data/dataset_alphafold.txt) to evaluate our method.

## Requirements and Environment
- Python == 3.8
- [HH-suite](https://github.com/soedinglab/hh-suite) for generating HHblits files from protein sequences (with the file suffix of .hhm)
- [Alphafold](https://github.com/deepmind/alphafold) for generating PDB files from protein sequences (with the file suffix of .pdb)
- Tensorflow == 2.7.0