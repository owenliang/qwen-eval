#!/bin/bash

wget 'https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/c-eval/ceval-exam.zip'
mkdir -p data/ceval
mv ceval-exam.zip data/ceval
cd data/ceval; unzip ceval-exam.zip
cd ../../