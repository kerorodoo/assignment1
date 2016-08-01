# assignment1
compile c-example:
    cd examples
    mkdir build
    cmake ..
    cmake --build .
compile python-example:
    apt-get install setuptools
    apt-get install boost-python-dev
    python setup.py install
