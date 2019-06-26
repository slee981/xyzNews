sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
wget http://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
rm Anaconda3-2019.03-Linux-x86_64.sh
pip3 install tensorflow keras pandas
sudo apt install -y jupyter-notebook