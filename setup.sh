pip install -e .

read -p "Path of the directory where datasets are stored and read: " dir
echo "REFERENCE_DIR = '$dir/reference_traverses/'" >> ./src/settings.py
echo "QUERY_DIR= '$dir/query_traverses/'" >> ./src/settings.py
