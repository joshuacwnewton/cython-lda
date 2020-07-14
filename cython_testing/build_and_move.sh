#!/usr/bin/env bash

# This build script automates rebuilding, and can be run before experiment.py

source ../venv/bin/activate

rm -rf ../src/build/
rm -rf ../src/cy_lda.c
rm -rf ../src/cy_lda*.so

python3 cy_setup.py build_ext --inplace

mv build/ ../src/build/
mv cy_lda.c ../src/cy_lda.c
mv cy_lda*.so ../src/cy_lda.so

deactivate