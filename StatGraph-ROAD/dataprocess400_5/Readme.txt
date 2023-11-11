This file contains the code for data processing and generated data is placed in the file "Dataset400_5".

Let TAG represent the type (source) of the data, TAG∈{1,2,3,4,5}

Each abnormal data will generate normal data set 0_TAG and abnormal data set 1_TAG, and all 0_TAG data will together constitute normal data set 0.

The data is sliced as follows:
                                                  train           val            test
Deal_correlated_signal_attack_masquerade            1              2               3
Deal_max_speedometer_attack_masquerade              1              2               3
Deal_reverse_light_off_attack_masquerade            1              2               3
Deal_reverse_light_on_attack_masquerade             1              2               3
Deal_max_engine_coolant_temp_attack_masquerade     70%            20%             10%


Each of the four types of attack data "Deal_correlated_signal_attack_masquerade", "Deal_max_speedometer_attack_masquerade", "Deal_reverse_light_off_attack_masquerade" and "Deal_reverse_light_on_attack_masquerade", have three files, corresponding to train, val and test. While attack data "Deal_max_engine_coolant_temp_attack_masquerade" has only one file, which is first identified as train file to generate normal data of "0_TAG" and abnormal data of "TAG". And then these two parts of data ("0_TAG" and "TAG") are divided into train, val and test in the proportion of 70%, 20% and 10% respectively.

It is worth noting that both the number of nodes (nnodes) and batchsize (batchsize) can be set by yourself. For example, we set nnodes=400 and batchsize=5.


