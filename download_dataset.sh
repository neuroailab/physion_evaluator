local_folder='/ccn2/u/rmvenkat/data/physion_data_download'

mkdir -p $local_folder
cd $local_folder
#scenarios='dominoes link roll support contain collide drop'
scenarios='dominoes'
sets='readout test'
for set in $sets
do
  mkdir -p $set
  cd $set
  for scenario in $scenarios
  do
    echo 'Downloading '$scenario'_'$set
    wget https://physion-model-$set-set-test.s3.us-east-2.amazonaws.com/$scenario.zip
  done
  cd ..
done
echo 'Downloaded all files'

echo 'Unzipping files'
for set in $sets
do
  cd $set
  for scenario in $scenarios
  do
    echo 'Unzipping '$scenario'_'$set
    unzip $scenario.zip
  done
  cd ..
done

echo 'Done unzipping files'

#delete zip files
for set in $sets
do
  cd $set
  for scenario in $scenarios
  do
    echo 'Deleting '$scenario'_'$set
    rm $scenario.zip
  done
  cd ..
done
