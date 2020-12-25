pip install -e .

echo "Downloading and extracting model weights"
cd src/feature_extraction
mkdir models
cd models
wget https://github.com/cedrusx/deep_features/releases/download/model_release_2/hfnet_vino_480x640_nms1r1.zip
wget https://github.com/cedrusx/deep_features/releases/download/model_release_1/hfnet_tf.zip
unzip hfnet_vino_480x640_nms1r1.zip
unzip hfnet_tf.zip
rm *.zip
cd ../../..

read -p "Path of the directory where datasets are stored and read: " data_dir
echo "DATA_DIR = '$data_dir'" >> ./src/settings.py
read -p "Path of the directory where results are to be stored: " results_dir
echo "RESULTS_DIR = '$results_dir'" >> ./src/settings.py
