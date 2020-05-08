# ConceptNet Discriminator

Discriminate concepts with attributes with ConceptNet5's Knowledge Graph. This tool uses ML and other technics to determinate whether two words can be discriminated by an attribute.

## Attribution

This work is based in [ConceptNet Knowledge Graph by Robyn Speer, Joshua Chin, and Catherine Havasi](https://github.com/commonsense/conceptnet5) and uses [Conceptnet-lite](https://github.com/ldtoolkit/conceptnet-lite) python module to navigate the graph.

Please refer to the following paper in further work.

> Robyn Speer, Joshua Chin, and Catherine Havasi. 2017. "ConceptNet 5.5: An Open Multilingual Graph of General Knowledge." In proceedings of AAAI 31.

## Usage

Before using this tool following requirements must be fullfilled:

- At least 15Gb of free space in your drive. [Conceptnet-lite](https://github.com/ldtoolkit/conceptnet-lite) automaticaly downloads a 11Gb database copy of ConceptNet.
- These python modules must be installed, you can install then with `pip install <module_name>`:
  - `conceptnet-lite`
  - `numpy`
  - `mongodb-community`
  - `sklearn`
  - `keras`
  - `nltk`
