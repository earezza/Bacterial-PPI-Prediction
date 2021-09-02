#!/bin/bash
args=("$@")
proteins=$(ls -A1 ${args[0]})
mkdir blastpssm/
mkdir blastout/
for p in $proteins
do
  	psiblast -db "${args[1]}" -evalue 0.001 -query "${args[0]}""${p}" -out_ascii_pssm blastpssm/${p}.pssm -out blastout/${p}.output -num_iterations 3
        cat blastpssm/${p}.pssm | sed '1,3d' | tac | sed '1,6d' | tac | cut -d' ' -f8- | column -t | tr ' ' '\t' | tr -s '\t' > ${args[0]}"${p}.mtx"
        rm blastout/${p}.output
        rm blastpssm/${p}.pssm
done

find "${args[0]}" -depth -name "*.txt.mtx" -exec sh -c 'f="{}"; mv -- "$f" "${f%.txt.mtx}"' \;
rm -r blastpssm
rm -r blastout
