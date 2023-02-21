A PyTorch implementation of hierarchical mixture-of-experts and budding trees.

## Getting started

- Clone the repo
  ```bash
  git clone https://github.com/oir/budtree.git
  cd budtree
  ```
- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
- Train HMOE on mnist
  ```bash
  python mnist.py
  ```
  In its first time this will attempt to download MNIST for you. If you have limited internet access
  or behind proxies you can download the following two files yourself and put in the root directory
  of the project (next to `mnist.py`):
  - https://pjreddie.com/media/files/mnist_train.csv
  - https://pjreddie.com/media/files/mnist_test.csv

  Run `python mnist.py --help` to see more options.
