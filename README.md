DQN requirement

apt-get install swig

git clone https://github.com/pybox2d/pybox2d
cd pybox2d
python setup.py build
python setup.py install

apt-get install -y xvfb

pip install \
    gym==0.21 \
    gym[box2d]==0.21 \
    pytorch-lightning==1.6.0 \