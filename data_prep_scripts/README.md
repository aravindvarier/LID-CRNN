## This folder contains all the files which are required to prepare the dataset.

# *IMPORTANT: All scripts in this folder have to be run from one folder above for them to function properly.*

The dataset used is the Voxforge dataset. The script currently downloads 5 Languages which is what the model is trained on as well. These are:
1. English
2. Spanish
3. German
4. Italian
5. French

Here is a short description of each script:

1. **download-data.sh**: The main download script which downloads and extracts the audio archives from the voxforge website. It also creates the audio folder with each language in a separate folder.
2. **extract\_tgz.sh**: Helper script called by the previous download script to unzip files.
3. **wav2png.py**: Converts the audio files to spectrograms and stores them as png files using the sox tool.
4. **png\_cleanup.py**: Makes all audio spectrograms equal size so they can be conveniently read in batches while training.
5. **csv\_file\_creator.py**: Creates csv files which are used by the dataloader script.

These scripts should be in the order in which they were described above.

# *Note: One can also run the main.sh script which will run all these scripts in one go. Currently, running main.sh just runs all the above programs with default command line arguments. To be added: functionality for command line arguments to main.sh*
