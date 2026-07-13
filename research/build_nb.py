"""Build a self-contained Kaggle notebook from WORKING_NOTE.md.
Every ![alt](figures/xxx.png) is replaced by an inline base64 <img>, so the notebook
renders identically on Kaggle with no external files. Splits into markdown cells per '## ' heading.
"""
import base64, json, re, os
ROOT='d:/ROGII'
md=open(f'{ROOT}/WORKING_NOTE.md',encoding='utf-8').read()

def embed(m):
    alt,path=m.group(1),m.group(2)
    fp=os.path.join(ROOT,path)
    b=base64.b64encode(open(fp,'rb').read()).decode()
    return (f'<div align="center"><img src="data:image/png;base64,{b}" '
            f'style="max-width:100%;height:auto;" alt="{alt}"/></div>')
md=re.sub(r'!\[([^\]]*)\]\(([^)]+\.png)\)', embed, md)

# split into cells: keep the title block, then one cell per top-level '## ' section
parts=re.split(r'(?m)^(?=## )', md)
cells=[]
for p in parts:
    p=p.strip('\n')
    if not p.strip(): continue
    cells.append({"cell_type":"markdown","metadata":{},"source":[l+'\n' for l in p.split('\n')]})

nb={"cells":cells,
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.10"}},
    "nbformat":4,"nbformat_minor":5}
open(f'{ROOT}/working_note_kaggle.ipynb','w',encoding='utf-8').write(json.dumps(nb,ensure_ascii=False,indent=1))
words=len(re.sub(r'<img[^>]+>','',md).split())
print('cells',len(cells),'approx words',words,'size(MB)',round(os.path.getsize(f"{ROOT}/working_note_kaggle.ipynb")/1e6,2))
