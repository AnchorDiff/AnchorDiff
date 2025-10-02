# AnchorDiff User Guide

1️⃣ Data Preparation
Download dataset and environment:

url:  https://pan.baidu.com/s/1eetuDUPcX1xk2FhBNH_wGA?pwd=xpki 

PWD: xpki 

**File Introduction**

- **AnchorDiff.zip**: Dataset compressed file
- **environment.yml**: Conda environment dependencies
- **mamba.zip**: Complete Conda environment
- **anchordiff_image.tar**: Docker image


2️⃣ Environment Setup
```
pip install -r requirements.txt
```
3️⃣ Quick Start
```
python ./core/match_relese/5_match_detail.py

python ./core/match_relese/6_read_result.py
```
4️⃣ Full Pipeline

Execute scripts 0-6 in sequence:
```
python ./core/match_relese/0_xxx.py
...
python ./core/match_relese/6_read_result.py
```


# AnchorDiff Docker Image Usage Guide

We have placed a Docker image that can be quickly executed at the URL below.

url:  https://pan.baidu.com/s/1eetuDUPcX1xk2FhBNH_wGA?pwd=xpki 

PWD: xpki 

FileName:  anchordiff_image.tar

## 1. Load Docker Image
```bash
docker load -i anchordiff_image.tar
```

## 2. Run Container
```bash
docker run -it --name anchordiff_container \
  -v /path/to/host/data:/AnchorDiff/data \
  anchordiff_image /bin/bash
```
Replace /path/to/host/data with your actual host directory path

## 3. Execute Pipeline Inside the Container

### Quick Start

```bash
conda activate mamba
cd /AnchorDiff
python ./core/match_relese/quick_start.py
```


### Full Pipeline (Execute scripts 0-6 in sequence)

```bash
conda activate mamba
cd /AnchorDiff
python ./core/match_relese/0_xxx.py
python ./core/match_relese/1_xxx.py
python ./core/match_relese/2_xxx.py
python ./core/match_relese/3_xxx.py
python ./core/match_relese/4_xxx.py
python ./core/match_relese/5_xxx.py
python ./core/match_relese/6_read_result.py
```




# ! Notice !
- Data processing/encoding tools contain closed-source components
- Related tools will be released in future updates

