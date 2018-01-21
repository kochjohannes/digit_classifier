#!/bin/bash
echo "Starting the setup, this will take approx. 10 min"
python3 download_and_train.py;
echo "Removing downloaded files."
rm mnist_train.csv; rm mnist_test.csv;
echo "Setup done.";
