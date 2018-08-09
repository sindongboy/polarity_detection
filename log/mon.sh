#!/bin/bash

while : 
do
	clear 
	cat ./polarity.log | grep "processing" | wc -l
	sleep 1
done
