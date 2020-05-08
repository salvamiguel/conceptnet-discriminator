# ConceptNet Discriminator

Discriminate concepts with attributes using ConceptNet5's Knowledge Graph. This tool uses ML and other technics to determinate whether two words can be discriminated by an attribute. For instance, given the words `apple`, `banana` and the attribute `red` this tool returns `True` because `apple` is `related_to` `red` while `banana` is not. This task was part of [SemEval-2018](https://www.aclweb.org/anthology/S18-1117.pdf), labeled as "Task 10: Capturing Discriminative Attributes".

## Attribution

This work is based in [ConceptNet Knowledge Graph by Robyn Speer, Joshua Chin, and Catherine Havasi](https://github.com/commonsense/conceptnet5) and uses [Conceptnet-lite](https://github.com/ldtoolkit/conceptnet-lite) python module to navigate the graph.

Please refer to the following paper in further work.

> Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge." In proceedings of AAAI 31. [Check it here](https://arxiv.org/pdf/1612.03975v2.pdf).

## Usage

### Requirements

Before using this tool following requirements must be fullfilled:
- Of course, `python3.7` has to be installed in your system though `python3.6` should be enough.
- At least 15Gb of free space in your drive. [Conceptnet-lite](https://github.com/ldtoolkit/conceptnet-lite) automaticaly downloads a zipped copy of ConceptNet's database and when unzipped can take up to 11Gb of disk space.
- These python modules must be installed, you can install them typing `pip install <module_name>` in your terminal or just run `pip install -r requirements.txt` to install them all:
  - `conceptnet-lite`
  - `numpy`
  - `mongodb-community`
  - `sklearn`
  - `keras`
  - `nltk`
  - `pymongo`
- **`mongodb` server must be running while executing ConceptNet Discriminator without auth**.
- While you can train yourself, is not recomended. Pre-trained weigths are used by default.

**Notice**: First time run can take several minutes while databases and weights are being downloaded from the Internet. 

### Command line
//TODO
### Web server
//TODO

