### 1. Create env

```
conda create --name atlas-env python=3.11 -y
```

### 2. Install atlas & olympus

```
git clone https://github.com/aspuru-guzik-group/atlas.git
cd atlas
pip install -e .
pip install -r requirements.txt

git clone -b olympus-atlas --single-branch https://github.com/aspuru-guzik-group/olympus.git
cd olympus
pip install -e .
```

### 3. Results

|             | 600  | 650  | 700  | 750  | 800  | 850  | 900  |
| :---------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Atlas-LSPR  |  13  |  5   |  5   |  9   |  5   |  6   |  6   |
| Astar-LSPR  |  3   |  2   |  2   |  7   |  7   |  5   |  4   |
| Atlas-FWHM  |  13  |  13  |  13  |  14  |  11  |  10  |  14  |
| Astar-FWHM  |  12  |  10  |  3   |  16  |  3   |  10  |  10  |
| Atlas-RATIO |  6   |  9   |  3   |  8   |  3   |  10  |  9   |
| Astar-RATIO |  10  |  3   |  3   |  5   |  3   |  9   |  9   |

