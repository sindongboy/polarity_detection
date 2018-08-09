#!/bin/bash


if [[ $# -ne 1 ]]; then 
	echo "usage: $0 [model.tar.gz]"
	exit 1
fi
rm -rf model

hadoop fs -rm -r /11st/regression/model

tar zxvf $1 

hadoop fs -put model /11st/regression/

hadoop fs -ls /11st/regression/model/

rm -rf model
