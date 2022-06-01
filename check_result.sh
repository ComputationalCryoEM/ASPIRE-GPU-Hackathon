#!/bin/bash

RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
NO_COLOR=$(tput sgr0)

n=$1
cmp_file=$2
orig_file='/tigress/cl5072/gpu_hackathon/ASPIRE-GPU-Hackathon/J_sync_vec_n'$n'.npy'
echo 'Comparing '$cmp_file' to '$orig_file'.'
if cmp -s $cmp_file $orig_file; then
    echo -e "${GREEN}Match.${NO_COLOR}"
else
    echo -e "${RED}No match.${NO_COLOR}"
fi


