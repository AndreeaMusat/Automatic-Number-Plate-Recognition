#!/bin/bash

for i in `seq 1 40`; do
	python3 main.py input/in$i.jpg
done