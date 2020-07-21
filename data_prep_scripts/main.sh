
if [ -d "data/audio_data" ]; then
    echo "The data folder already exists. Do you want to delete and download it again? Enter Yes to download. Enter anything else to not and proceed with training."
    read reply
    if [ $reply = "yes" ] || [ $reply = "Yes" ] || [ $reply = 'Y' ] || [ $reply = 'y' ]; then
        rm -rf "data/audio_data"
        if [ $# -lt 1 ]; then
            ./data_prep_scripts/download-data.sh 15000 english french spanish german italian
        else
            ./data_prep_scripts/download-data.sh $1 english french spanish german italian
        fi  
    fi
else
    if [ $# -lt 1 ]; then
        ./data_prep_scripts/download-data.sh 15000 english french spanish german italian
    else
        ./data_prep_scripts/download-data.sh $1 english french spanish german italian
    fi
fi

python "data_prep_scripts/wav2png.py"

python "data_prep_scripts/png_cleanup.py"

python "data_prep_scripts/csv_file_creator.py"
