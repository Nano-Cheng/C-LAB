#### 1. Create env

```
conda create --name atlas-env python=3.11 -y
```

#### 2. Install atlas & olympus

```
git clone https://github.com/aspuru-guzik-group/atlas.git
cd atlas
pip install -e .
pip install -r requirements.txt

git clone -b olympus-atlas --single-branch https://github.com/aspuru-guzik-group/olympus.git
cd olympus
pip install -e .
```

