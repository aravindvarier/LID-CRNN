#Needs to be run from one directory above
if [ $# -lt 1 ]; then
  echo "Usage: $0 <#downloads> <list of languages>"
  echo "Example: ./download-data.sh 15000 english french"
  exit 1
fi

NUM_DOWN=$1
shift
TEMP_DIR=tmp
OUTPUT_DIR=audio_data_np

if [ ! -d $OUTPUT_DIR ]
then
	mkdir $OUTPUT_DIR
fi


source `dirname $0`/voxforge_download_urls
for lang in $@
do
	eval VOXFORGE_DATA_URL=\$$lang

	if [ "${VOXFORGE_DATA_URL}x" = "x" ]; then
	  echo "Can't find url for language $lang"
	  exit 1
	fi

	ZIPS=zipped_index_$lang

	curl $VOXFORGE_DATA_URL | grep -o '<a .*href=.*>' | sed -e 's/<a /\n<a /g' | sed -e 's/<a .*href=['"'"'"]//' -e 's/["'"'"'].*$//' -e '/^$/ d' | grep tgz$  > $ZIPS

	if [ ! -d $lang ]; then
  		mkdir $lang
	fi

	if [ ! -d $OUTPUT_DIR/$lang ]; then
		mkdir $OUTPUT_DIR/$lang
	fi

	for ZIP in $(cat $ZIPS)
	do
		if [ `ls -U $lang | wc -l` -le $NUM_DOWN ];then
	   		URL=$VOXFORGE_DATA_URL/$ZIP
	   		wget --no-verbose -q --directory-prefix=$TEMP_DIR $URL
	  		`dirname $0`/extract_tgz.sh $TEMP_DIR/$ZIP $lang
			python data_prep.py $lang
			rm $lang/*  #deletes all wav files
			rm -rf $TEMP_DIR/* #deletes all folders and files in tmp which contains the zipped folders and unzipped folders
		fi
	done	
	rm $ZIPS
	rm -rf $lang

done

rm -rf $TEMP_DIR
