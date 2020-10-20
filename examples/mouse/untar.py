import os
import sys
import subprocess

print('extracting data')

dirname = 'th-1/data/'
files = os.listdir(dirname)
files = [f for f in files if 'Mouse' in f]

print('searching', files)

for f in files:
    if ('.tar.gz' in f):
        print('\nextracting', f)
        base = f[:-7]
        subprocess.call(['tar -zxvf ' + dirname + f], shell=True)
        subprocess.call([
            'rm ' + base + '/' + '*.fet.* ' + base + '/' + '*.spk.* ' + base +
            '/' + '*eeg'
        ],
                        shell=True)
        subprocess.call(['rm ' + dirname + f], shell=True)

subprocess.call(
    ['tar -zxvf ' + dirname + 'PositionFiles.tar.gz --directory ' + dirname],
    shell=True)
subprocess.call(['rm ' + dirname + 'PositionFiles.tar.gz'], shell=True)
subprocess.call(
    ['mv ' + dirname + 'PositionFiles/Mouse28/Mouse28-140313/Mouse*ang .'],
    shell=True)
subprocess.call(['rm -r ' + dirname + 'PositionFiles'], shell=True)
