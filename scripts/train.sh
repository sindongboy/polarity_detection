#!/bin/bash



hadoop fs -rm -r /11st/regression/model

spark-submit --class com.skplanet.nlp.driver.PolarityDetectionTrainer --master local[*] --deploy-mode client  --driver-class-path ../../../resource/nlp-resource/config:../../../resource/nlp-resource/resource --conf spark.executor.memory=16G --conf spark.driver.memory=16G ../target/polarity-detection-1.0.0-jar-with-dependencies.jar \
	-m /11st/regression/model -w /11st/w2v/w2vModel -t /11st/regression/train/sentiments -d 500
