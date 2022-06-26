# space-hhblits
Evaluation on different combinations of AlphaFold2 (coordinates) and HHblits.

## Download data
We provide the dataset used in this study,  you can download [dataset_alphafold.txt](https://github.com/Liuzhe30/space-hhblits/blob/main/data/dataset_alphafold.txt) to evaluate our method.

## Requirements and Environment
- Python == 3.8
- Tensorflow == 2.7.0
- [HH-suite](https://github.com/soedinglab/hh-suite) for generating HHblits files from protein sequences (with the file suffix of .hhm)
- [Alphafold](https://github.com/deepmind/alphafold) for generating PDB files from protein sequences (with the file suffix of .pdb)
- [DSSP](https://github.com/cmbi/dssp) for generating DSSP files from pdb files (with the file suffix of .dssp)

## Citation
Please cite the following paper for using this code: 
```
Z. Liu et al., "Will AlphaFold2 Be Helpful in Improving the Accuracy of Single-sequence PPI Site Prediction?," 2022 10th International Conference on Bioinformatics and Computational Biology (ICBCB), 2022, pp. 18-23, doi: 10.1109/ICBCB55259.2022.9802490.
```