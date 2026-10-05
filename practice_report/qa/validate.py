import json,platform,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from core.experiment_service import ExperimentService,ExperimentOptions
from core.dataset import TextDataset
root=Path('practice_report/qa');checks=[]
for ext in ('txt','csv','json'):
 d=TextDataset(f'datasets/demo_texts.{ext}').load();checks.append([f'Імпорт {ext.upper()}',len(d.texts)==50,f'{len(d.texts)} документів'])
s=ExperimentService(str(root/'runs'))
o=ExperimentOptions(input_path='datasets/demo_texts.csv',vectorizers=('tfidf',),algorithms=('kmeans','parallel_kmeans','hierarchical','dbscan'),n_clusters=5,runs=3,lemmatize=False)
r,a=s.run(o)
r.to_json(root/'smoke_results.json',orient='records',indent=2,force_ascii=False)
checks.append(['Повний цикл без лематизації',len(r)==4 and 'error' not in r.columns,'4 комбінації, 3 повтори'])
r2,a2=s.load_saved_experiment(s.run_output_dir);checks.append(['Відновлення',len(r2)==len(r),'4 рядки без перерахунку'])
try:TextDataset('/tmp/nonexistent_practice_274.txt').load();checks.append(['Відсутній файл',False,''])
except FileNotFoundError:checks.append(['Відсутній файл',True,'FileNotFoundError'])
p=root/'bad.csv';p.write_text('wrong\nhello\n')
try:TextDataset(str(p)).load();checks.append(['Відсутня колонка',False,''])
except ValueError:checks.append(['Відсутня колонка',True,'ValueError'])
meta={'checks':checks,'python':sys.version.split()[0],'platform':platform.platform(),'artifacts':a,'options':o.__dict__}
(root/'validation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(r[['algorithm','silhouette','f1_macro','clustering_seconds','noise_documents']].to_string(index=False))
