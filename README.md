# STIX-MWA
A repo for all things related to finding solar flares in MWA data.

## Installation

This repo is compatible with Python versions 3.8 and 3.10 (because of the latest version of CASA, version 6.6).

```bash
# clone project
git clone https://github.com/PredragMatavulj/STIX-MWA
cd your-repo-name
```

#### Pip

```bash
# create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

To install mantaray (https://github.com/MWATelescope/manta-ray-client), do:
 - git clone https://github.com/ICRAR/manta-ray-client.git
 - cd manta-ray-client
 - python3 setup.py install

## Needed environment variables

You need to set the following environment variables in the above created `.env` file. Refer
to `.env.example` for an example.