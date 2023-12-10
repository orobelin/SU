# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi
vbnbnvbn
# User specific environment and startup programs

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH


module load CDO/1.9.5-nsc1-intel-2018a-eb
module load ncview/2.1.7-nsc1
module load netCDF/4.4.1.1-HDF5-1.8.19-nsc1-intel-2018a-eb
module load NCO/4.7.9-nsc5
module use /proj/bolinc/shared/software/modules
#module load ElmerIce/058434f-intel-2018b-eb
#module load ElmerIce/2021-01-19-intel-2018b-eb
#module load ElmerIce/2022-03-02-intel-2018b-eb
module load ElmerIce/2022-03-02-gcc-2018a-eb
module load MUMPS/5.2.1-metis-nsc1-gcc-2018a-eb
export LD_LIBRARY_PATH=$EBROOTMUMPS/lib:$LD_LIBRARY_PATH
#module load buildenv-intel/2018b-eb
module load buildenv-gcc/2018a-eb
module load ParaView/5.4.1-nsc1-gcc-2018a-eb
module load Python/3.8.3-anaconda-2020.07-extras-nsc1
module load gmsh/4.8.1-gcc-2018a-eb  

alias sq='squeue -u x_jamba'  #--long
alias home='cd /proj/bolinc/users/x_jamba/'
alias kg='cd /proj/bolinc/users/x_jamba/RUNS_KG'
alias lh='ls -lh'


export ELMER_HOME="/proj/bolinc/shared/software/apps/ElmerIce/2022-03-02/gcc-2018b-eb"
#export ELMER_HOME="/proj/bolinc/shared/software/apps/ElmerIce/2021-01-19/intel-2018b-eb"
#export ELMER_HOME="/proj/bolinc/shared/software/apps/ElmerIce/2022-03-02/intel-2018b-eb"

export PATH="$PATH:$ELMER_HOME/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ELMER_HOME/share/elmersolver/lib"



export LD_LIBRARY_PATH=/proj/bolinc/users/x_jamba/extra_solvers:$LD_LIBRARY_PATH
export RPATH=/proj/bolinc/users/x_jamba/extra_solvers:$RPATH


