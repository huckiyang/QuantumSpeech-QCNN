#!/bin/bash
cat asr_q_tr.z* > tmp1.zip && unzip -FF tmp1.zip
cat asr_x_tr.z* > tmp2.zip && unzip -FF tmp2.zip
unzip asr_q_val.zip
