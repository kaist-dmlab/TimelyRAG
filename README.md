# TimelyRAG: Semantic-Temporal Hybrid Retrieval for Time-Critical Question Answering with Evolving Documents

This is the official implementation of **_TimelyRAG_**


## Public Datasets
|       Name       |         Domain        | #Docs | #Queries |  Temporal Type    | 
| :--------------: | :-------------------: | :---: | :------: | :---------------: | 
|        Law       |   Legal Regulations   | 30,000 |   3,000  | Overlapping-Evolving | 
|    University    |   Academic Policies   | 30,000 |   3,000  | Overlapping-Evolving | 
| Company Policies |  Corporate Guidelines | 30,000 |   3,000  | Overlapping-Evolving |
| Terms of Service |  Service Policies | 30,000 |   3,000  | Overlapping-Evolving | 

## Requirements and Installations
- [Node.js](https://nodejs.org/en/download/): 16.13.2+
- [Anaconda 4](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniconda 3](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.11.5 (Recommend Anaconda)
- Ubuntu 18.04.6 LTS
- pytorch >= 2.1.2

## Configuration
All hyperparameters and paths are managed in config.py.
- Parameter options
```bash
--exp-dir: Experiment output directory (string)
--gpu: GPU device id (int)
--datasets: Comma-separated JSONL dataset paths (list)
```

