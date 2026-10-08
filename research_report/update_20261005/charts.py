from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics import renderSVG
from pathlib import Path
import json, math
P=Path(__file__).parent
s=json.loads((P/'summary.json').read_text());vs=json.loads((P/'vector_summary.json').read_text())
V=['tfidf','word2vec','fasttext','doc2vec','sentence_bert'];VN=['TF-IDF','Word2Vec','FastText','Doc2Vec','Sentence-BERT']
A=['kmeans','parallel_kmeans','hierarchical','dbscan','minibatch','birch','spectral','affinity','hdbscan'];AN=['K-Means','Parallel K-Means','Ward','DBSCAN','MiniBatch','BIRCH','Spectral','Affinity','HDBSCAN']
from reportlab.lib.colors import HexColor,Color,white,black
blue=HexColor('#315D82'); orange=HexColor('#BD692C')
def tx(d,x,y,t,size=10,anchor='start',color=black): d.add(String(x,y,str(t),fontName='Helvetica',fontSize=size,textAnchor=anchor,fillColor=color))
def get(v,a):return next(r for r in s if r['vectorizer']==v and r['algorithm']==a)
def save(d,n):renderSVG.drawToFile(d,str(P/(n+'.svg')))
d=Drawing(720,365)
for j,n in enumerate(VN):tx(d,220+j*100,342,n,10,'middle')
for i,a in enumerate(A):
 y=310-i*32;tx(d,158,y+9,AN[i],11,'end')
 for j,v in enumerate(V):
  z=get(v,a)['f1_macro_mean']; c=Color(1-.83*z/.85,1-.65*z/.85,1-.4*z/.85);d.add(Rect(173+j*100,y,95,29,fillColor=c,strokeColor=white));tx(d,220+j*100,y+9,f'{z:.3f}',11,'middle')
save(d,'quality')
d=Drawing(720,315)
for j,(v,n) in enumerate(zip(V,VN)):
 x=90+j*125;tx(d,x+20,26,n,10,'middle')
 for b,a in enumerate(['kmeans','parallel_kmeans','hierarchical','minibatch']):
  z=get(v,a)['clustering_seconds_median']*1000;d.add(Rect(x+(b-1.5)*18,48,16,z*7.5,fillColor=[blue,orange,HexColor('#689268'),HexColor('#999999')][b],strokeColor=None))
for y in range(0,31,5): d.add(Line(35,48+y*7.5,690,48+y*7.5,strokeColor=HexColor('#dddddd'),strokeWidth=.4));tx(d,30,45+y*7.5,y,9,'end')
for j,(n,c) in enumerate(zip(['K-Means','Parallel K-Means','Ward','MiniBatch'],[blue,orange,HexColor('#689268'),HexColor('#999999')])):d.add(Rect(65+j*165,293,9,9,fillColor=c,strokeColor=None));tx(d,80+j*165,293,n,10)
save(d,'time')
d=Drawing(720,275)
for i,v in enumerate(V):
 tx(d,135,211-i*38,VN[i],11,'end')
 for j,z in enumerate(get(v,'parallel_kmeans')['per_core_median']):
  c=Color(1,1-.7*z/100,1-.8*z/100);d.add(Rect(150+j*45,199-i*38,43,34,fillColor=c,strokeColor=white));tx(d,171+j*45,211-i*38,f'{z:.0f}',10,'middle')
for j in range(12):tx(d,171+j*45,246,j+1,10,'middle')
save(d,'cores')
# Two outliers are shown on a linear scale, retaining the entire series.
import pandas as pd
f=pd.read_csv(P/'all_900_records.csv');d=Drawing(720,315)
for y in [0,50,100,150,200,250,300]:d.add(Line(45,45+y*.7,695,45+y*.7,strokeColor=HexColor('#dddddd'),strokeWidth=.5));tx(d,38,42+y*.7,y,9,'end')
for v,c in zip(V,[blue,orange,HexColor('#58875D'),HexColor('#86668F'),HexColor('#222222')]):
 g=f[(f.vectorizer==v)&(f.algorithm=='parallel_kmeans')];pts=[(50+(r.series-1)*33,45+r.clustering_seconds*1000*.7) for r in g.itertuples()]
 for a,b in zip(pts,pts[1:]):d.add(Line(*a,*b,strokeColor=c,strokeWidth=1))
 for x,y in pts:d.add(Circle(x,y,2,fillColor=c,strokeColor=None))
for i in range(20):tx(d,50+i*33,29,i+1,9,'middle')
for j,(n,c) in enumerate(zip(VN,[blue,orange,HexColor('#58875D'),HexColor('#86668F'),HexColor('#222222')])):d.add(Rect(35+j*138,287,8,8,fillColor=c,strokeColor=None));tx(d,48+j*138,287,n,9)
save(d,'parallel_series')
