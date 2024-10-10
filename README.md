### 📋 **A chemical autonomous robotic platform for the synthesis of nanoparticles**

<div>
<span class="author-block">
  Fan Gao<sup>a†</sup>
</span>,
<span class="author-block">
  Hongqiang Li<sup>b†</sup>
</span>,
<span class="author-block">
  Zhilong Chen<sup>a†</sup>
</span>,
<span class="author-block">
   Yunai Yi<sup>b</sup>
</span>,
<span class="author-block">
  Shihao Nie<sup>c</sup>
</span>,
<span class="author-block">
  Zeming Liu<sup>c*</sup>
</span>,
<span class="author-block">
  Yuanfang Guo<sup>c</sup>
</span>,
<span class="author-block">
  Shumin Liu<sup>a</sup>
</span>,
<span class="author-block">
  Qizhen Qin<sup>a</sup>
</span>,
<span class="author-block">
  Zhengjian Li<sup>a</sup>
</span>,
<span class="author-block">
  Lisong Zhang<sup>d</sup>
</span>,
<span class="author-block">
  Han Hu<sup>d</sup>
</span>,
<span class="author-block">
  Cunjin Li<sup>d</sup>
</span>,
<span class="author-block">
  Liang Yang<sup>e</sup>
</span>,
<span class="author-block">
  Guangxu Chen<sup>a*</sup>
</span>
</div>

- **a.** School of Environment and Energy, State Key Laboratory of Luminescent Materials and Devices, Guangdong Provincial Key Laboratory of Atmospheric Environment and Pollution Control, South China University of Technology

- b. Zhuhai Fengze Information Technology Co., Ltd., Zhuhai 519000, China.

- c. School of Computer Science and Engineering, Beihang University, Beijing, 100191, China

- d. Guangzhou Inlab, Guangzhou 510530, China.

- e. School of Artificial Intelligence, Hebei University of Technology, Tianjin, 300401, China

  F. G., H. L. and Z.L. contributed to this work equally.

Email: cgx08@scut.edu.cn; zmliu@buaa.edu.cn



## ⚙️ Usage

### Installation
```
git clone https://github.com/Nano-Cheng/C-LAB.git
cd C-LAB
pip install -r requirements.txt
```

### Summary
```
python main.py --mode 'summary' --dataset_paper './dataset_paper' --dataset_summary './dataset_summary'
```
### Query-Localize

```
python main.py --mode 'query_localize' --query_text 'Au NRs' --dataset_summary 'dataset_summary' --similarity_top_k 3 --similarity_cutoff 0.7
```

### Query-Extract

```
python main.py --mode 'query_extract' --query_text 'Au NRs' --extract_pdf_path '{your pdf path}'
```



# 💗 Acknowledgements
We appreciate the financial support from the National Nature Science Foundation of China (21971070), Guangdong Innovative and Entrepreneurial Research Team Program (2019ZT08L075), Guangdong Pearl River Talent Program (2019QN01L159), Science and Technology Program of Guangzhou, China (202103040002). We thank professor Li Xia from South China University of technology for his helpful discussion on the algorithm in this work.

### 🛎 Citation
If you find our work helpful for your research, please cite:
```bib
```
