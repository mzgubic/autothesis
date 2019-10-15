for file in $(ls raw_html/)
do
    echo $file
    cat raw_html/${file} | grep "pdf" > skimmed_html/${file}.txt
done
