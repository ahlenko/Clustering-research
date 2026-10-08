from pathlib import Path
import json,hashlib,datetime
import pandas as pd
import numpy as np
OUT=Path(__file__).parent
V=['tfidf','word2vec','fasttext','doc2vec','sentence_bert']; A=['kmeans','parallel_kmeans','hierarchical','dbscan','minibatch','birch','spectral','affinity','hdbscan']
VN=['TF-IDF','Word2Vec','FastText','Doc2Vec','Sentence-BERT']; AN=['K-Means','Parallel K-Means','Ward','DBSCAN','MiniBatch','BIRCH','Spectral','Affinity','HDBSCAN']
rows=[];series=[];hashes={}
for i,d in enumerate(sorted(Path('reports/experiments').iterdir()),1):
 c=json.loads(next(d.glob('configuration*')).read_text()); r=json.loads(next(d.glob('results*.json')).read_text()); m=json.loads((d/'manifest.json').read_text())
 assert len(r)==45 and {(x['vectorizer'],x['algorithm']) for x in r}=={(v,a) for v in V for a in A}
 assert all('error' not in x for x in r)
 for x in r: rows.append(dict(x,series=i,experiment=d.name))
 start=datetime.datetime.strptime(d.name,'experiment_%Y%m%d_%H%M%S');end=datetime.datetime.fromisoformat(m['created_at'])
 series.append(dict(series=i,folder=d.name,start=start.isoformat(),end=end.isoformat(),envelope=(end-start).total_seconds()))
 for f in list(d.glob('results*.json'))+list(d.glob('configuration*.json')):hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
f=pd.DataFrame(rows); w=f[f.series>1];f.to_csv(OUT/'all_900_records.csv',index=False)
summary=[]
for v in V:
 for a in A:
  g=w[(w.vectorizer==v)&(w.algorithm==a)];first=f[(f.series==1)&(f.vectorizer==v)&(f.algorithm==a)].iloc[0]
  z={'vectorizer':v,'algorithm':a}
  for k in ['clustering_seconds','clustering_memory_kb','cpu_percent','f1_macro','ari','silhouette','davies_bouldin','calinski_harabasz','f1_micro','nmi','v_measure','clusters_found','noise_documents']:
   for n,val in [('mean',g[k].mean()),('median',g[k].median()),('sd',g[k].std(ddof=1)),('min',g[k].min()),('max',g[k].max()),('q1',g[k].quantile(.25)),('q3',g[k].quantile(.75)),('first',first[k])]:z[k+'_'+n]=float(val)
  z['per_core_median']=np.median(np.stack(g.cpu_per_core_percent),axis=0).tolist()
  summary.append(z)
s=pd.DataFrame(summary);s.to_json(OUT/'summary.json',orient='records',force_ascii=False,indent=2)
vector=[]
for v in V:
 g=w[w.vectorizer==v].drop_duplicates('series');first=f[(f.series==1)&(f.vectorizer==v)].iloc[0]
 z={'vectorizer':v}
 for k in ['vectorization_seconds','vectorization_memory_kb']:
  for n,val in [('mean',g[k].mean()),('median',g[k].median()),('sd',g[k].std(ddof=1)),('min',g[k].min()),('max',g[k].max()),('first',first[k])]:z[k+'_'+n]=float(val)
 vector.append(z)
pd.DataFrame(vector).to_json(OUT/'vector_summary.json',orient='records',indent=2)
for item in series:
 g=f[f.series==item['series']];item['vector_seconds']=g.drop_duplicates('vectorizer').vectorization_seconds.sum();item['cluster_seconds']=g.clustering_seconds.sum();item['measured_sum']=item['vector_seconds']+item['cluster_seconds']
(OUT/'series.json').write_text(json.dumps(series,indent=2));(OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
print('ROWS',len(f),'QUALITY UNCHANGED',all(f.groupby(['vectorizer','algorithm'])[k].nunique().max()==1 for k in ['f1_macro','ari','silhouette','nmi','f1_micro','v_measure','davies_bouldin','calinski_harabasz']))
print('SERIES',series)
print('PARALLEL OUTLIERS',f[(f.algorithm=='parallel_kmeans')&(f.clustering_seconds>.05)][['series','vectorizer','clustering_seconds','cpu_percent']].to_dict('records'))
print('PAIR')
for v in V:
 a=s[(s.vectorizer==v)&(s.algorithm=='kmeans')].iloc[0];b=s[(s.vectorizer==v)&(s.algorithm=='parallel_kmeans')].iloc[0]
 print(v,'ratio',b.clustering_seconds_median/a.clustering_seconds_median,'memory',b.clustering_memory_kb_median/a.clustering_memory_kb_median,'F1diff',b.f1_macro_mean-a.f1_macro_mean,'ARIdiff',b.ari_mean-a.ari_mean,'mean median max',b.clustering_seconds_mean,b.clustering_seconds_median,b.clustering_seconds_max)
print('INITIAL CLUSTER RATIOS',sorted([(x['clustering_seconds_first']/x['clustering_seconds_median'],x['vectorizer'],x['algorithm']) for x in summary],reverse=True)[:8])

print('under .1 sec',sum(f.clustering_seconds<.1),'maxcpu',f.cpu_percent.max())
