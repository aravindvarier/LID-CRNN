rm -rf *.csv

rm -rf audio_data_np

rm -rf zipped_index_*

if [ $# -lt 1 ]; then
    ./download-data.sh 15000 english french spanish german italian
else
    ./download-data.sh $1 english french spanish german italian
fi

python csv_file_creator.py

python train.py

python test.py