#!/bin/bash

Rscript MGC-test2.R
Rscript MGC-test3.R

sudo pip3 install runipy
jupyter nbconvert --to notebook --execute NDD_Sprint1.ipynb --output mynotebook.ipynb

