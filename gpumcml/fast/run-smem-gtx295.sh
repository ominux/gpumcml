#!/bin/bash

PROG=gpumcml.sm_13
INPUT_FILE=input/skin/7layerskin_600nm_100M.mci

NZ=500
NR=200
NV=3584    # max number of voxels to be cached

MAX_IZ=${NZ}

while [[ ${MAX_IZ} -gt 0 ]]; do

    # Customize the header file for MAX_IZ.
    GREP_STR="s/##MAX_IZ##/${MAX_IZ}/g"
    sed -i.bak ""${GREP_STR}"" gpumcml_kernel.h

    # Determine the max IR we can try.
    MAX_IR=$(( NV / MAX_IZ ))
    if [[ ${MAX_IR} -gt ${NR} ]]; then
        MAX_IR=${NR}
    fi

    while [[ ${MAX_IR} -gt 0 ]]; do

        # Check if MAX_IR * MAX_IZ is a multiple of 32.
        MAX_NV=$(( MAX_IZ * MAX_IR ))
        REM=$(( MAX_NV % 32 ))
        if [[ ${REM} -eq 0 ]]; then

            # Customize the header file for MAX_IR.
            GREP_STR="s/##MAX_IR##/${MAX_IR}/g"
            sed -i.bak1 ""${GREP_STR}"" gpumcml_kernel.h
            make ${PROG}

            echo "(IR,IZ) = (${MAX_IR},${MAX_IZ})"

            {

            echo
            echo "=========== (IR,IZ) = (${MAX_IR},${MAX_IZ}) ==============="

            ./${PROG} ${INPUT_FILE}
            ./${PROG} ${INPUT_FILE}
            ./${PROG} ${INPUT_FILE}

            } >> smem.out

            # Restore the header file.
            mv gpumcml_kernel.h.bak1 gpumcml_kernel.h
        fi

        MAX_IR=$(( MAX_IR - 1 ))
    done

    # Restore the header file.
    mv gpumcml_kernel.h.bak gpumcml_kernel.h

    # Decrement MAX_IZ by 4.
    MAX_IZ=$(( MAX_IZ - 4 ))
done

exit 0

