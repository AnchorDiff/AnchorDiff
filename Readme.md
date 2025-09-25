# AnchorDiff User Guide

1️⃣ Data Preparation
Download dataset and organize directory:

url:

```
/
├── core/
├── data/
│   ├── libvdiff/
├   |   |── Anchor/
│   |   ├── *
├   ├── OSS_lib/
├── Readme.md
```



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

# ! Notice !
- Data processing/encoding tools contain closed-source components
- Related tools will be released in future updates

