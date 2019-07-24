# select ubuntu 18.04 minimal 
#
# gcsfuse does not work with ^18.10

sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
sudo apt install -y screen
sudo apt install -y vim

# wget http://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh
# bash Anaconda3-2019.03-Linux-x86_64.sh
# rm Anaconda3-2019.03-Linux-x86_64.sh
# sudo apt install -y jupyter-notebook

pip3 install tensorflow keras pandas numpy twilio sklearn tqdm

export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

mkdir data && chmod 777 data/
gcsfuse word-embed data

echo 'alias mount="gcsfuse word-embed data"' >> .bashrc 
echo 'alias c="clear"' >> .bashrc
source .bashrc

# jupyter notebook --generate-config

# echo "c = get_config()" >> /home/smlee_981/.jupyter/jupyter_notebook_config.py
# echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/smlee_981/.jupyter/jupyter_notebook_config.py
# echo "c.NotebookApp.open_browser = False" >> /home/smlee_981/.jupyter/jupyter_notebook_config.py
# echo "c.NotebookApp.port = 5000" >> /home/smlee_981/.jupyter/jupyter_notebook_config.py
# echo 'alias jn="jupyter notebook --no-browser --port=5000"' >> .bashrc 

mkdir results && chmod 777 results/
