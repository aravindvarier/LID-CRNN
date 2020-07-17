
if [ -d "audio_data_np" ]; then
    echo "The data folder already exists. Do you want to delete and download it again? Enter Yes to download. Enter anything else to not and proceed with training."
    read reply
    if [ $reply = "yes" ] || [ $reply = "Yes" ] || [ $reply = 'Y' ] || [ $reply = 'y' ]; then
        rm -rf audio_data_np
        if [ $# -lt 1 ]; then
            ./download-data.sh 15000 english french spanish german italian
        else
            ./download-data.sh $1 english french spanish german italian
        fi  
    fi
else
    if [ $# -lt 1 ]; then
        ./download-data.sh 15000 english french spanish german italian
    else
        ./download-data.sh $1 english french spanish german italian
    fi
fi

rm -rf *.csv #removing any remnants from previous runs. don't want lines to get appended to previous csv files
python "csv_file_creator.py"
python "train.py"
python "test.py"






