martinize2 -f IGG.pdb -x dppc-md.pdb -elastic true -bonds-from name -merge H,K,L,M
gmx editconf -f DPPC-em.gro -d 1.0  -o dppc-md.gro 
gmx editconf -f dppc-md.gro -c -o dppc-md.gro -bt triclinic -box 10 10 10
gmx insert-molecules -f DPPC_10x10.gro -ci backbone.gro -o dppc-md2.gro -nmol 1 -try 5000 -selrpos atom

gmx grompp -p top.top -f minimization.mdp -c dppc-md2.gro -o minimization-vac1.tpr
gmx mdrun -deffnm minimization-vac1 -v
gmx grompp -p top.top -f minimization.mdp -c minimization-vac1.gro -o minimization-vac1.tpr
gmx insert-molecules -f minimization-vac1.gro -ci CL.pdb -o dppc-md2_CL.gro -nmol 1248
gmx insert-molecules -f dppc-md2_CL.gro -ci NA.pdb -o dppc-md2_CLNA.gro -nmol 385
gmx insert-molecules -f dppc-md2_CLNA.gro -ci CL.pdb -o dppc-md2_ION.gro -nmol 385

gmx grompp -p top.top -f minimization.mdp -c dppc-md2_CL.gro -o minimization-vac2.tpr
gmx mdrun -deffnm minimization-vac2 -v
gmx solvate -cp minimization-vac2.gro -cs water.gro -radius 0.21  -o solvated.gro -p top.top
gmx grompp -p top.top -c solvated.gro -f minimization.mdp -o minimization.tpr
gmx mdrun -deffnm minimization -v
gmx grompp -f martini_md.mdp -c minimization.gro -p top.top -o dppc-md.tpr -maxwarn 2
gmx mdrun -s dppc-md.tpr -v -x dppc-md.xtc -c DPPC-md.gro 




#####plumed
gmx_plu grompp -f martini_md.mdp -c minimization.gro -p top.top -o dppc-md.tpr -maxwarn 2
gmx_plu mdrun -s dppc-md.tpr -v -x dppc-md.xtc -c DPPC-md.gro -plumed plumed.dat -nt 15

#include "martini_v3.0.0.itp" 
#include "martini_v3.0.0_ions_v1.itp" 
#include "water.itp"
volume  36745.6 nm^3