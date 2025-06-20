sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=EvSpecies   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ethanwhite@ufl.edu # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/blue/ewhite/everglades/EvergladesSpeciesModel/logs/EvergladesSpeciesModel_%j.out   # Standard output and error log
#SBATCH --error=/blue/ewhite/everglades/EvergladesSpeciesModel/logs/EvergladesSpeciesModel_%j.err
#SBATCH --gpus=1

ulimit -c 0
source activate ESM
python everglades_species.py
EOT
