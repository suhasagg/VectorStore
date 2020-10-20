# `vectorstore`

**Tired of your custom word/sentence vectors taking too much RAM!!**

**Scale it using Vector-Store**

>**Powered by LevelDB**

Not excited about having all your RAM eaten up?
Still want fast lookup?

## Key Features
* Easily initialize custom word/sentence embedding databases on disk.
* Lazy-load embeddings instead of reading the entire file on startup.
* LRU caching (1024 most recent lookups by default)
 
Open Vectorstore word embeddings DB

```Python
>>> from emstore import Emstore
>>> e = Emstore('~/topicDataLevelDB')
>>> e['modi']
[0.27204,
 -0.06203,
 -0.1884,
 0.023225,
 -0.018158,
 0.0067192,
 ...
]
```
## Installing

#### Linux

You'll need leveldb
```
apt-get update && apt-get install -y \
    gcc g++ libxml2-dev libxslt1-dev zlib1g-dev
apt-get install -y libleveldb1 libleveldb-dev
```
Requirements and vectorstore:
```
pip install -r requirements.txt
python setup.py install
```
#### Sample Test Script
```
ScaleTopicVectors.py
Takes parameters 1 or 2 
```

Also see included docker-compose and Dockerfile.

#### OS X

This can be tricky due to dependencies.

I was able to get everything to work by using this procedure:

1. install leveldb with homebrew: `brew install leveldb`
2. clone plyvel git repository and install from source:
```shell
git clone https://github.com/wbolster/plyvel.git
cd plyvel
make
pip install .
3. installing emstore with `pip install .` or `python setup.py install`
```

## Dependencies

This software is made possible with special thanks to:
- [Emstore](https://github.com/MichaMucha/emstore)
💚🧡💜💙💖😀😊😍🌹🌹🌹


## Contributing

This is an early release. 
Your feedback and use cases will be appreciated.

feel free to contribute improvements as well. Some ideas:
 - Support for other KV stores like BadgerDB, BoltDB
 
#### License: MIT

