sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
pip3 install -r requirements.txt 

# install gcsfuse for connection to word embeddings
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

# connect to word embeddings
gcsfuse word-embed ./app/word_embedding