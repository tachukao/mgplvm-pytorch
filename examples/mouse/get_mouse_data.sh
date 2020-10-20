#!/bin/bash

# must provide crcns account and pw as arguments

#change permissions
chmod u+x download.sh

#set user and pw
TEXT="crcns_username='"$1"'\ncrcns_password='"$2"'\n"
printf $TEXT > crcns-account.txt

#download data
./download.sh th-1

#untar and throw away the stuff we don't need
python untar.py

#bin the data for use with mGPLVM
python bin_data.py

