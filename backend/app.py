import io,json,os,uuid,threading,html
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI,UploadFile,File,HTTPException,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,Response,HTMLResponse
from pydantic import BaseModel,Field
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor,export_text
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.inspection import permutation_importance
ROOT=Path(os.getenv('DATA_DIR',str(Path(__file__).parent/'data'/'projects'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
REQ=['Machine_ID','Machine_Category','Maintenance_Date','Downtime_Hours','Production_Volume','Failure_Count']
OUT=['Downtime_Hours','Production_Volume','Failure_Count']
OPT=['Observation_Date','Period_Start','Period_End','Maintenance_Event_ID','Maintenance_Type','Operating_Hours','Planned_Production_Hours','Maintenance_Performed','Planned_Maintenance_Hours']
app=FastAPI(title='Industrial Maintenance Effectiveness Analyzer')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(','),allow_methods=['*'],allow_headers=['*'])
LOCK=threading.RLock()
class Setup(BaseModel):
 row_meaning:str='unconfirmed'
 period_days:int=Field(1,ge=1,le=366)
 date_format:str='%Y-%m-%d'
 event_meaning:str='unconfirmed'
 maintenance_type:str='unspecified'
 window_days:int=Field(30,ge=1,le=180)
 minimum_coverage:float=Field(.8,ge=.5,le=1)
 minimum_records:int=Field(10,ge=2)
 minimum_category_machines:int=Field(4,ge=3)
 high_quantile:float=Field(.75,ge=.5,le=.95)
 improvement_threshold:float=Field(5,ge=0,le=100)
def serial(v):
 if isinstance(v,dict): return {str(k):serial(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [serial(x) for x in v]
 if isinstance(v,np.integer): return int(v)
 if isinstance(v,(float,np.floating)): return float(v) if np.isfinite(v) else None
 if v is pd.NaT or v is pd.NA: return None
 if isinstance(v,pd.Timestamp): return v.isoformat()
 return v
def save(p,v):
 t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(serial(v),indent=2),encoding='utf-8');t.replace(p)
def folder(pid):
 try: uuid.UUID(pid)
 except ValueError: raise HTTPException(404,'Unknown project')
 p=ROOT/pid
 if not (p/'metadata.json').exists(): raise HTTPException(404,'Unknown project')
 return p
def meta(p): return json.loads((p/'metadata.json').read_text(encoding='utf-8'))
def parse(raw):
 if not raw.strip(): raise HTTPException(422,'CSV is empty')
 try: d=pd.read_csv(io.BytesIO(raw),dtype=str)
 except Exception as e: raise HTTPException(422,f'Cannot parse CSV: {e}')
 known={c.lower():c for c in REQ+OPT};d.columns=[known.get(c.strip().replace(' ','_').lower(),c.strip()) for c in d.columns]
 if d.columns.duplicated().any(): raise HTTPException(422,'Column names collide after normalization')
 if d.empty: raise HTTPException(422,'CSV has no data rows')
 return d.map(lambda x:x.strip() if isinstance(x,str) else x).replace('',np.nan)
def clean(p):
 s=Setup(**meta(p)['setup']);d=parse((p/'original.csv').read_bytes());missing=set(REQ)-set(d)
 if missing: raise HTTPException(422,'Missing required columns: '+', '.join(sorted(missing)))
 original=len(d);issues=[]
 def note(msg,mask):
  if mask.sum(): issues.append({'issue':msg,'records':int(mask.sum())})
 dup=d.duplicated();note('Exact duplicates removed',dup);d=d.loc[~dup].copy()
 invalid=d.Machine_ID.isna()|d.Machine_Category.isna();note('Missing identifier/category: rows excluded',invalid)
 for c in OUT+[c for c in ['Operating_Hours','Planned_Production_Hours','Planned_Maintenance_Hours'] if c in d]:
  n=pd.to_numeric(d[c],errors='coerce');bad=(d[c].notna()&n.isna())|(n<0)|~np.isfinite(n.fillna(0))
  if c=='Failure_Count': bad|=n.notna()&(n%1!=0)
  note(c+': invalid values set missing',bad);note(c+': missing values',d[c].isna());d[c]=n.mask(bad)
 for c in ['Maintenance_Date','Observation_Date','Period_Start','Period_End']:
  if c in d:
   v=pd.to_datetime(d[c],format=s.date_format,errors='coerce');note(c+': invalid dates set missing',d[c].notna()&v.isna());d[c]=v
 if 'Maintenance_Performed' in d:
  v=pd.to_numeric(d.Maintenance_Performed,errors='coerce');note('Invalid event indicator',~v.isin([0,1]));d['Maintenance_Performed']=v.where(v.isin([0,1]))
 cats=d.groupby('Machine_ID').Machine_Category.nunique();bad=d.Machine_ID.isin(cats[cats>1].index);note('Inconsistent category: rows excluded',bad);invalid|=bad
 if {'Period_Start','Period_End'}<=set(d):
  bad=d.Period_End<d.Period_Start;note('Invalid period ordering: rows excluded',bad);invalid|=bad
  # Calendar dates use inclusive endpoints; never infer missing boundaries.
  hours=((d.Period_End-d.Period_Start).dt.days+1)*24
  bad=(hours>0)&(d.Downtime_Hours>hours);note('Downtime exceeds explicit calendar period: set missing',bad);d.loc[bad,'Downtime_Hours']=np.nan
 if s.row_meaning in ['daily','period']:
  bad=d.Downtime_Hours>24*s.period_days;note('Downtime exceeds confirmed duration: set missing',bad);d.loc[bad,'Downtime_Hours']=np.nan
 d=d.loc[~invalid].copy()
 if s.row_meaning=='daily' and 'Observation_Date' in d:
  bad=d.duplicated(['Machine_ID','Observation_Date'],keep=False)&d.Observation_Date.notna();note('Multiple machine-day rows: timing disabled',bad);d.loc[bad,'Observation_Date']=pd.NaT
 outliers={}
 for c in OUT:
  q1,q3=d[c].quantile([.25,.75]);outliers[c]=int((d[c]>q3+1.5*(q3-q1)).sum())
 q={'original_records':original,'clean_records':len(d),'duplicates_removed':int(dup.sum()),'issues':issues,'valid_outcomes':{c:int(d[c].count()) for c in OUT},'outliers_for_investigation':outliers,'preview':d.head(12).to_dict('records')}
 d.to_csv(p/'cleaned.csv',index=False);save(p/'quality.json',q);return d,q
def dataset(p):
 d=pd.read_csv(p/'cleaned.csv',dtype={'Machine_ID':str,'Machine_Category':str})
 for c in ['Observation_Date','Maintenance_Date','Period_Start','Period_End']:
  if c in d: d[c]=pd.to_datetime(d[c],errors='coerce')
 return d
def events(p,d,s):
 if (p/'events.csv').exists():
  e=parse((p/'events.csv').read_bytes());e.Maintenance_Date=pd.to_datetime(e.Maintenance_Date,format=s.date_format,errors='coerce')
 elif s.event_meaning=='indicator' and 'Maintenance_Performed' in d: e=d[d.Maintenance_Performed==1].copy()
 elif s.event_meaning=='event_date': e=d.copy()
 else: return pd.DataFrame(columns=['Machine_ID','Maintenance_Date'])
 keys=['Machine_ID','Maintenance_Date']
 if 'Maintenance_Event_ID' in e and e.Maintenance_Event_ID.notna().all(): keys+=['Maintenance_Event_ID']
 return e.dropna(subset=['Machine_ID','Maintenance_Date']).drop_duplicates(keys)
def paired(d,e,s):
 if s.row_meaning!='daily' or 'Observation_Date' not in d or not d.Observation_Date.notna().any(): return {'available':False,'reason':'Before/after comparison unavailable: observation timing is missing. Confirm daily rows and supply Observation_Date. Interval rows need daily disaggregation.','pairs':[],'excluded':{}}
 if e.empty: return {'available':False,'reason':'Confirmed maintenance events are required.','pairs':[],'excluded':{}}
 pairs=[];excluded={'overlapping':0,'coverage':0};w=s.window_days
 for mid,ee in e.groupby('Machine_ID'):
  dates=sorted(ee.Maintenance_Date.tolist());obs=d[d.Machine_ID==mid]
  for event in dates:
   if sum(abs((event-other).days)<=2*w for other in dates)>1: excluded['overlapping']+=1;continue
   pre=obs[(obs.Observation_Date>=event-pd.Timedelta(days=w))&(obs.Observation_Date<event)]
   post=obs[(obs.Observation_Date>event)&(obs.Observation_Date<=event+pd.Timedelta(days=w))]
   ns=[int(x.loc[x.Downtime_Hours.notna(),'Observation_Date'].nunique()) for x in [pre,post]]
   if min(ns)<w*s.minimum_coverage or min(ns)/max(ns)<.9: excluded['coverage']+=1;continue
   a,b=pre.Downtime_Hours.mean(),post.Downtime_Hours.mean()
   r={'machine':mid,'event':str(event.date()),'pre':a,'post':b,'absolute_change':b-a,'reduction_pct':(a-b)/a*100 if a else None,'pre_n':ns[0],'post_n':ns[1],'pre_coverage':ns[0]/w,'post_coverage':ns[1]/w}
   for c in ['Failure_Count','Production_Volume']:
    ns=[int(x[c].count()) for x in [pre,post]];ok=min(ns)>=w*s.minimum_coverage and min(ns)/max(ns)>=.9
    r[c]={'pre':pre[c].mean() if ok else None,'post':post[c].mean() if ok else None,'pre_n':ns[0],'post_n':ns[1]}
   pairs.append(r)
 return {'available':bool(pairs),'reason':None if pairs else 'No events meet non-overlap and coverage requirements.','pairs':pairs,'excluded':excluded,'window_days':w,'convention':'Event day excluded; all overlapping event windows excluded. Both windows require configured coverage and valid-day count ratio >=90%. Values are per observed day.'}
def read_evaluation(p):
 f=p/'evaluation.json';return json.loads(f.read_text()) if f.exists() else {'status':'not_trained'}
def analysis(p,machine=None,category=None,start=None,end=None):
 s=Setup(**meta(p)['setup']);full=dataset(p);e=events(p,full,s);d=full.copy()
 if machine: d=d[d.Machine_ID==machine]
 if category: d=d[d.Machine_Category==category]
 for val,is_start in [(start,True),(end,False)]:
  if val:
   if 'Observation_Date' not in d: raise HTTPException(422,'Observation dates unavailable')
   try: ts=pd.Timestamp(val)
   except Exception: raise HTTPException(422,'Invalid filter date')
   d=d[d.Observation_Date>=ts] if is_start else d[d.Observation_Date<=ts]
 e=e[e.Machine_ID.isin(d.Machine_ID)];horizon=None
 if s.row_meaning=='daily' and 'Observation_Date' in d and d.Observation_Date.notna().any():
  lo,hi=d.Observation_Date.min(),d.Observation_Date.max();horizon=(hi-lo).days+1;e=e[(e.Maintenance_Date>=lo)&(e.Maintenance_Date<=hi)]
 comp=paired(d,e,s);rows=[];comparable=s.row_meaning in ['daily','period']
 for mid,g in d.groupby('Machine_ID'):
  count=int((e.Machine_ID==mid).sum());days=int(g.Observation_Date.nunique()) if 'Observation_Date' in g else None;pp=[x for x in comp['pairs'] if x['machine']==mid]
  reductions=[x['reduction_pct'] for x in pp if x['reduction_pct'] is not None]
  rows.append({'machine':mid,'category':str(g.Machine_Category.iloc[0]),'n':len(g),'downtime':g.Downtime_Hours.mean(),'downtime_total':g.Downtime_Hours.sum(min_count=1),'failures':g.Failure_Count.mean(),'production':g.Production_Volume.mean(),'valid_n':{c:int(g[c].count()) for c in OUT},'events':count,'frequency':count/horizon*30 if horizon else None,'observed_days':days,'coverage':days/horizon if horizon and days is not None else None,'paired_events':len(pp),'reduction_pct':np.mean(reductions) if reductions else None,'flag':'Insufficient evidence'})
 for r in rows:
  peers=[x for x in rows if x['category']==r['category'] and min(x['valid_n']['Downtime_Hours'],x['valid_n']['Failure_Count'])>=s.minimum_records and (not horizon or x['coverage']>=s.minimum_coverage)]
  if not comparable or len(peers)<s.minimum_category_machines or r not in peers: continue
  high=r['downtime']>=np.quantile([x['downtime'] for x in peers],s.high_quantile) or r['failures']>=np.quantile([x['failures'] for x in peers],s.high_quantile)
  r['flag']='Investigate recurring failures' if high else 'Routine monitoring'
  if horizon and r['coverage']>=s.minimum_coverage and high and r['events']>0 and r['events']>=np.quantile([x['events'] for x in peers],s.high_quantile) and r['reduction_pct'] is not None and r['reduction_pct']<=s.improvement_threshold: r['flag']='Review maintenance effectiveness'
 cats=[]
 for cat,g in d.groupby('Machine_Category'):
  rr=[x for x in rows if x['category']==cat]
  cats.append({'category':cat,'machines':len(rr),'n':len(g),'downtime':np.nanmean([r['downtime'] for r in rr]),'variation_sd':np.nanstd([r['downtime'] for r in rr]),'failures':np.nanmean([r['failures'] for r in rr]),'production':np.nanmean([r['production'] for r in rr]),'events':sum(r['events'] for r in rr),'coverage':np.mean([r['coverage'] for r in rr]) if horizon else None,'paired_events':sum(r['paired_events'] for r in rr),'valid_n':{c:int(g[c].count()) for c in OUT}})
 confirmed=s.event_meaning!='unconfirmed' or (p/'events.csv').exists()
 metrics={'machines':d.Machine_ID.nunique(),'categories':d.Machine_Category.nunique(),'records':len(d),'events':len(e) if confirmed else None,'downtime':d.Downtime_Hours.sum(min_count=1),'failures':d.Failure_Count.sum(min_count=1),'production':d.Production_Volume.sum(min_count=1),'paired_events':len(comp['pairs']),'valid_n':{c:int(d[c].count()) for c in OUT}}
 limitations=['Observational associations do not establish causation. Difficult machines may receive more maintenance.','Category averages weight machines equally; within-machine means weight records equally. Missing outcomes are excluded, never zero-filled.','Events deduplicated by machine/date/ID if IDs are complete; otherwise same-day events cannot be distinguished.','Frequency uses the shared calendar horizon; observed-day coverage is reported. Missing event logs may undercount events.','Priorities use category quantiles and paired evidence; recent-trend and maintenance-gap rules are not used.']
 if not comparable: limitations.append('Period semantics unconfirmed: per-record descriptions only; comparisons require comparable durations.')
 insights=[]
 def add(title,evidence,limitation): insights.append({'title':title,'evidence':evidence,'limitation':limitation})
 if len(d):
  add('Recorded downtime burden',f"{metrics['downtime']:.2f} hours; {metrics['valid_n']['Downtime_Hours']} valid records, {metrics['machines']} machines.",'Totals depend on observation exposure.')
  valid=[r for r in rows if pd.notna(r['downtime_total'])]
  if valid:
   top=max(valid,key=lambda x:x['downtime_total']);add('Largest recorded contribution',f"{top['machine']}: {top['downtime_total']:.2f} hours; {top['valid_n']['Downtime_Hours']} valid records.",'Raw contribution is not a rate or causal ranking.')
  if cats and comparable:
   top=max(cats,key=lambda x:x['downtime']);add('Category comparison',f"{top['category']}: {top['downtime']:.2f} mean downtime hours per confirmed period; {top['machines']} machines, {top['valid_n']['Downtime_Hours']} valid records.",'Workload and machine condition may confound comparisons.')
  changes=[x['absolute_change'] for x in comp['pairs']]
  change_text=f" Mean paired change: {np.mean(changes):+.3f} downtime hours per observed day (after minus before)." if changes else ''
  add('Paired comparison support',f"{len(comp['pairs'])} accepted events; {sum(comp['excluded'].values())} excluded."+change_text,comp.get('reason') or comp['convention'])
  supported=[r for r in rows if pd.notna(r['downtime']) and r['valid_n']['Downtime_Hours']>=s.minimum_records and (not horizon or r['coverage']>=s.minimum_coverage)]
  association=''
  if confirmed and comparable and len(supported)>=5 and np.std([r['events'] for r in supported])>0 and np.std([r['downtime'] for r in supported])>0:
   corr=np.corrcoef([r['events'] for r in supported],[r['downtime'] for r in supported])[0,1]
   association=f" Machine-level Pearson association with mean downtime: r={corr:.3f}, n={len(supported)} machines."
  add('Maintenance event evidence',(f"{len(e)} confirmed events across {len(rows)} machines." if confirmed else f"No established events across {len(rows)} machines.")+association,'Count is not a rate without a horizon; pooled association is confounded by category, workload and machine condition. Type: '+s.maintenance_type+'.')
  flagged=[r for r in rows if r['flag'] in ['Investigate recurring failures','Review maintenance effectiveness']]
  add('Review candidates',f"{len(flagged)} of {len(rows)} machines meet review rules.",f"Minimum {s.minimum_records} valid records and {s.minimum_category_machines} category peers; do not reduce maintenance solely from poor outcomes.")
 ev=read_evaluation(p)
 if ev.get('status')=='complete': add('Held-out model performance',f"MAE {ev['test']['mae']:.2f} h versus baseline {ev['baseline']['mae']:.2f} h; n={ev['test_n']}.",'Full unfiltered dataset. Predictive patterns and category EDA can differ because the model conditions on prior measurements.')
 recommendations={
  'Recorded downtime burden':('Audit downtime coding and missing daily records.','Production analyst','Valid outcome coverage and downtime hours/day'),
  'Largest recorded contribution':('Inspect repeat failure modes and operating loads on the leading contributor.','Reliability engineer','Repeat failures and downtime hours/day'),
  'Category comparison':('Check comparable workloads, ages and production units before comparing categories.','Production engineer','Machine-weighted downtime under matched workloads'),
  'Paired comparison support':('Improve event spacing and complete daily logs; review supported paired changes with an engineer.','Maintenance planner','Eligible paired events and daily coverage'),
  'Maintenance event evidence':('Audit preventive/corrective event types and unique event IDs; investigate workload confounding.','Maintenance planner','Event-log completeness and confirmed event counts'),
  'Review candidates':('Review recurring faults, maintenance quality and machine condition; do not reduce schedules solely from poor outcomes.','Reliability engineer','Repeat failures and paired downtime change'),
  'Held-out model performance':('Validate on a later untouched operating period and compare against both simple baselines.','Data analyst','Future-period MAE and baseline MAE')}
 actions=[]
 for i in insights:
  action,owner,kpi=recommendations[i['title']]
  actions.append({'finding':i['title'],'action':action,'owner':owner,'priority':'High' if 'candidate' in i['title'] else 'Medium','kpi':kpi,'validation':'Equal-exposure follow-up with comparable machines; account for workload and missing observations.'})
 result=serial({'metrics':metrics,'machines':rows,'categories':cats,'comparison':comp,'events_available':confirmed,'horizon_days':horizon,'comparable':comparable,'unit':'hours / observed day' if s.row_meaning=='daily' else 'hours / record','limitations':limitations,'insights':insights,'actions':actions,'filters':{'machine':machine,'category':category,'start':start,'end':end},'criteria':s.model_dump()});save(p/'analysis.json',result);return result

def features(d,e,s):
 cols=['Machine_Category','Production_Volume','Failure_Count'];mode='explanatory'
 if s.row_meaning in ['daily','period'] and 'Observation_Date' in d and d.Observation_Date.notna().any():
  mode='forecast';d=d.dropna(subset=['Observation_Date']).sort_values(['Machine_ID','Observation_Date']).copy();d=d.loc[~d.duplicated(['Machine_ID','Observation_Date'],keep=False)].copy();g=d.groupby('Machine_ID');d['target']=g.Downtime_Hours.shift(-1);d['target_date']=g.Observation_Date.shift(-1)
  for c in OUT: d['previous_'+c]=d[c]
  d['historical_events']=0.;d['days_since_maintenance']=np.nan
  for mid,g in d.groupby('Machine_ID'):
   dates=np.array(sorted(e.loc[e.Machine_ID==mid,'Maintenance_Date']),dtype='datetime64[ns]');current=g.Observation_Date.to_numpy(dtype='datetime64[ns]');counts=np.searchsorted(dates,current,side='right');d.loc[g.index,'historical_events']=counts
   if len(dates): d.loc[g.index,'days_since_maintenance']=np.where(counts>0,(current-dates[np.maximum(counts-1,0)])/np.timedelta64(1,'D'),np.nan)
  cols=['Machine_Category']+['previous_'+c for c in OUT]
  if s.event_meaning!='unconfirmed' or len(e): cols+=['historical_events','days_since_maintenance']
  d=d[(d.target_date-d.Observation_Date).dt.days==s.period_days]
 else: d=d.copy();d['target']=d.Downtime_Hours
 return d.dropna(subset=['target']),cols,mode
def scores(y,p): return {'mae':mean_absolute_error(y,p),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':r2_score(y,p) if len(y)>1 and np.var(y)>0 else None}
def decision_rules(tree,names):
 result=[]
 def walk(node,conditions):
  t=tree.tree_
  if t.children_left[node]<0:
   result.append({'conditions':' AND '.join(conditions) or 'All training records','estimated_hours':float(t.value[node].ravel()[0]),'training_support':int(t.n_node_samples[node])})
   return
  name=names[t.feature[node]];threshold=t.threshold[node]
  walk(t.children_left[node],conditions+[f'{name} <= {threshold:.3f}'])
  walk(t.children_right[node],conditions+[f'{name} > {threshold:.3f}'])
 walk(0,[])
 return sorted(result,key=lambda r:r['training_support'],reverse=True)[:5]
def train(p,version):
 try:
  s=Setup(**meta(p)['setup'])
  if s.row_meaning not in ['daily','period','event']: raise ValueError('Confirm row semantics before training.')
  d=dataset(p);d,cols,mode=features(d,events(p,d,s),s)
  if len(d)<50: raise ValueError('At least 50 valid target records required.')
  if mode=='forecast':
   dates=sorted(d.target_date.unique())
   if len(dates)<10: raise ValueError('At least 10 distinct target dates required.')
   cutoff=dates[int(len(dates)*.8)];a=d.index[d.target_date<cutoff];b=d.index[d.target_date>=cutoff];split=f'Chronological holdout: targets before {pd.Timestamp(cutoff).date()} train; on/after test. Identical target dates stay together.'
  else:
   if d.Machine_ID.nunique()<5: raise ValueError('At least five machines required for grouped evaluation.')
   aa,bb=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(d,groups=d.Machine_ID));a=d.index[aa];b=d.index[bb];split='Machine-grouped holdout: generalization to unseen machines.'
  if len(a)<30 or len(b)<10: raise ValueError('Minimum train/test sample sizes are 30/10.')
  nums=[c for c in cols if c!='Machine_Category'];prep=ColumnTransformer([('numeric',SimpleImputer(strategy='median',keep_empty_features=True),nums),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),['Machine_Category'])]);model=Pipeline([('preprocess',prep),('tree',DecisionTreeRegressor(max_depth=4,min_samples_leaf=max(5,int(len(a)*.01)),random_state=42))])
  xtr,xte=d.loc[a,cols],d.loc[b,cols];ytr,yte=d.loc[a,'target'],d.loc[b,'target'];model.fit(xtr,ytr);dummy=DummyRegressor().fit(np.zeros((len(a),1)),ytr);perm=permutation_importance(model,xte,yte,n_repeats=5,random_state=42,scoring='neg_mean_absolute_error');tree=model.named_steps['tree']
  ev={'status':'complete','mode':mode,'target':'Downtime hours in the next observation period' if mode=='forecast' else 'Same-period downtime hours (explanatory; not forecasting)','split':split,'train_n':len(a),'test_n':len(b),'train':scores(ytr,model.predict(xtr)),'test':scores(yte,model.predict(xte)),'baseline':scores(yte,dummy.predict(np.zeros((len(b),1)))),'features':cols,'importance':[{'feature':c,'mae_increase':v,'std':z} for c,v,z in zip(cols,perm.importances_mean,perm.importances_std)],'tree':export_text(tree,feature_names=list(model.named_steps['preprocess'].get_feature_names_out())),'leaf_support':[{'node':int(i),'training_records':int(tree.tree_.n_node_samples[i]),'prediction':float(tree.tree_.value[i].ravel()[0])} for i in range(tree.tree_.node_count) if tree.tree_.children_left[i]<0],'ranges':{c:[float(xtr[c].min()),float(xtr[c].max())] for c in nums},'categories':sorted(xtr.Machine_Category.unique().tolist()),'version':version,'complexity':'Fixed depth 4; minimum leaf max(5, 1% of training records); no test-set tuning.','limitations':'Predictive associations, not causes or engineering limits. Same-period explanatory inputs may reflect downtime. Full dataset used regardless of filters.','train_machines':sorted(d.loc[a,'Machine_ID'].unique().tolist()),'test_machines':sorted(d.loc[b,'Machine_ID'].unique().tolist())}
  ev['rules']=decision_rules(tree,model.named_steps['preprocess'].get_feature_names_out())
  if mode=='forecast': ev.update(persistence_baseline=scores(yte,d.loc[b,'previous_Downtime_Hours'].fillna(xtr.previous_Downtime_Hours.median())),train_last_target=str(d.loc[a,'target_date'].max()),test_first_target=str(d.loc[b,'target_date'].min()))
  with LOCK:
   if meta(p)['version']!=version: return
   joblib.dump(model,p/'pipeline.joblib');save(p/'evaluation.json',ev)
 except Exception as exc:
  with LOCK:
   if meta(p)['version']==version: save(p/'evaluation.json',{'status':'failed','reason':str(exc)})
@app.get('/api/projects')
def projects(): return [meta(p) for p in ROOT.iterdir() if (p/'metadata.json').exists()]
def create(raw,name,confirmed=False):
 d=parse(raw);missing=set(REQ)-set(d)
 if missing: raise HTTPException(422,'Missing required columns: '+', '.join(sorted(missing)))
 pid=str(uuid.uuid4());p=ROOT/pid;p.mkdir();s=Setup(row_meaning='daily',event_meaning='indicator',maintenance_type='as recorded') if confirmed else Setup()
 save(p/'metadata.json',{'id':pid,'name':Path(name).name,'version':1,'setup':s.model_dump(),'source':'User-supplied synthetic dataset' if confirmed else 'User upload'});(p/'original.csv').write_bytes(raw);clean(p);return meta(p)
@app.post('/api/projects')
async def upload(file:UploadFile=File(...)):
 raw=await file.read(20*1024*1024+1)
 if len(raw)>20*1024*1024: raise HTTPException(413,'Maximum upload size 20 MB')
 return create(raw,file.filename or 'upload.csv')
@app.post('/api/projects/supplied')
def supplied():
 p=Path(__file__).parent/'data'/'supplied.csv'
 if not p.exists(): raise HTTPException(404,'Supplied dataset not installed')
 return create(p.read_bytes(),'industrial_manufacturing_synthetic.csv',True)
@app.get('/api/projects/{pid}')
def project(pid:str): return meta(folder(pid))
@app.put('/api/projects/{pid}/setup')
def setup(pid:str,s:Setup):
 if s.row_meaning not in ['unconfirmed','daily','period','event'] or s.event_meaning not in ['unconfirmed','indicator','event_date']: raise HTTPException(422,'Unsupported semantics')
 if s.row_meaning=='daily' and s.period_days!=1: raise HTTPException(422,'Daily rows require period_days=1')
 if s.date_format not in ['%Y-%m-%d','%d/%m/%Y','%m/%d/%Y']: raise HTTPException(422,'Unsupported explicit date format')
 p=folder(pid)
 with LOCK:
  m=meta(p);m['setup']=s.model_dump();m['version']+=1;save(p/'metadata.json',m);clean(p);save(p/'evaluation.json',{'status':'not_trained','reason':'Setup changed; prior model invalidated.'});(p/'analysis.json').unlink(missing_ok=True)
 return m
@app.post('/api/projects/{pid}/events')
async def upload_events(pid:str,file:UploadFile=File(...)):
 p=folder(pid);raw=await file.read(20*1024*1024+1)
 if len(raw)>20*1024*1024: raise HTTPException(413,'Maximum upload size 20 MB')
 e=parse(raw)
 if not {'Machine_ID','Maintenance_Date'}<=set(e): raise HTTPException(422,'Events require Machine_ID and Maintenance_Date')
 dates=pd.to_datetime(e.Maintenance_Date,format=meta(p)['setup']['date_format'],errors='coerce')
 if e.Machine_ID.isna().any() or dates.isna().any(): raise HTTPException(422,'Event identifiers/dates must be valid and nonempty')
 with LOCK:
  (p/'events.csv').write_bytes(raw);m=meta(p);m['version']+=1;save(p/'metadata.json',m);save(p/'evaluation.json',{'status':'not_trained'});(p/'analysis.json').unlink(missing_ok=True)
 return {'event_rows':len(e),'distinct_machine_dates':len(e.drop_duplicates(['Machine_ID','Maintenance_Date']))}
@app.get('/api/projects/{pid}/quality')
def quality(pid:str): return json.loads((folder(pid)/'quality.json').read_text())
@app.post('/api/projects/{pid}/clean')
def cleaning(pid:str): return serial(clean(folder(pid))[1])
@app.get('/api/projects/{pid}/analysis')
def get_analysis(pid:str,machine:str|None=None,category:str|None=None,start:str|None=None,end:str|None=None): return analysis(folder(pid),machine,category,start,end)
@app.get('/api/projects/{pid}/{section}')
def section(pid:str,section:str,machine:str|None=None,category:str|None=None,start:str|None=None,end:str|None=None):
 p=folder(pid)
 if section in ['evaluation','training-status']: return read_evaluation(p)
 if section not in ['before-after','frequency','priorities','insights']: raise HTTPException(404,'Unknown resource')
 a=analysis(p,machine,category,start,end);return a[{'before-after':'comparison','frequency':'machines','priorities':'machines','insights':'insights'}[section]]
@app.post('/api/projects/{pid}/train')
def training(pid:str,tasks:BackgroundTasks):
 p=folder(pid)
 with LOCK:
  if read_evaluation(p).get('status')=='training': raise HTTPException(409,'Training in progress')
  save(p/'evaluation.json',{'status':'training'});tasks.add_task(train,p,meta(p)['version'])
 return {'status':'training'}
@app.post('/api/projects/{pid}/predict')
def predict(pid:str,values:dict):
 p=folder(pid);ev=read_evaluation(p)
 if ev.get('status')!='complete' or ev.get('version')!=meta(p)['version']: raise HTTPException(409,'Train a compatible model first')
 warnings=[];record={}
 for c in ev['features']:
  v=values.get(c)
  if c=='Machine_Category':
   if not isinstance(v,str) or not v.strip(): raise HTTPException(422,'Machine_Category required')
   if v not in ev['categories']: warnings.append('Unseen machine category')
  else:
   try: v=float(v)
   except (TypeError,ValueError): raise HTTPException(422,f'{c} must be numeric')
   if not np.isfinite(v) or v<0: raise HTTPException(422,f'{c} must be finite and nonnegative')
   if ('Failure_Count' in c or c=='historical_events') and v%1: raise HTTPException(422,f'{c} must be an integer')
   lo,hi=ev['ranges'][c]
   if lo is not None and (v<lo or v>hi): warnings.append(c+' outside training range')
  record[c]=v
 model=joblib.load(p/'pipeline.joblib');return {'downtime_hours':float(model.predict(pd.DataFrame([record]))[0]),'warnings':warnings,'mode':ev['mode']}

def report(a,ev):
 esc=lambda x:html.escape(str(x))
 def table(rows):
  if not rows: return '<p>No supported results.</p>'
  keys=list(rows[0]);return '<table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(r.get(k,''))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
 def bars(rows,label,value):
  valid=[r for r in rows if isinstance(r.get(value),(float,int))];mx=max([r[value] for r in valid]+[1]);h=32*len(valid)+35
  return f'<svg role="img" aria-label="{esc(value)}" viewBox="0 0 900 {h}">'+''.join(f'<text x="0" y="{i*32+22}">{esc(r[label])}</text><rect x="220" y="{i*32+7}" width="{r[value]/mx*480}" height="20" fill="#0d9488"/><text x="740" y="{i*32+22}">{r[value]:.2f}</text>' for i,r in enumerate(valid))+'</svg>'
 def scatter(rows,x,y):
  valid=[r for r in rows if isinstance(r.get(x),(int,float)) and isinstance(r.get(y),(int,float))];mx=max([r[x] for r in valid]+[1]);my=max([r[y] for r in valid]+[1])
  return '<svg role="img" viewBox="0 0 900 350"><path d="M60 20V300H850" fill="none" stroke="#64748b"/><text x="350" y="340">'+esc(x)+'</text><text x="10" y="18">'+esc(y)+'</text>'+''.join(f'<circle cx="{60+r[x]/mx*760}" cy="{300-r[y]/my*260}" r="6" fill="{"#e07a3f" if r["flag"] not in ["Routine monitoring","Insufficient evidence"] else "#0d9488"}"><title>{esc(r)}</title></circle>' for r in valid)+'</svg><p>Orange: review candidates; teal: other machines. Axes start at zero; maxima '+f'{mx:.2f}, {my:.2f}</p>'
 comp=a['comparison'];pairs=comp['pairs']
 charts='<h2>1. Before/after downtime</h2>'+(bars([{'label':r['machine']+' '+r['event']+' '+k,'hours':r[k]} for r in pairs for k in ['pre','post']],'label','hours')+table(pairs) if pairs else '<p>'+esc(comp['reason'])+'</p><h3>Descriptive downtime by machine</h3>'+bars(a['machines'],'machine','downtime'))
 charts+='<h2>2. Downtime by category</h2>'+bars(a['categories'],'category','downtime')+table(a['categories'])
 charts+='<h2>3. Maintenance '+('frequency per 30 calendar days' if a['horizon_days'] else 'event count')+' versus downtime</h2>'+(scatter(a['machines'],'frequency' if a['horizon_days'] else 'events','downtime') if a['events_available'] else '<p>Confirmed events unavailable.</p>')
 charts+='<h2>4. Machine priority overview</h2>'+scatter(a['machines'],'failures','downtime')+table(a['machines'])
 return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Maintenance report</title><style>body{font:15px system-ui;margin:40px;color:#172b35}table{border-collapse:collapse;width:100%;font-size:11px;overflow-wrap:anywhere}td,th{border:1px solid #ddd;padding:6px;text-align:left}svg{width:100%}pre{white-space:pre-wrap}h2{margin-top:40px}</style><h1>Industrial Maintenance Effectiveness Analyzer</h1><p>Observational analysis | '+esc(a['unit'])+'</p><p>Filters: '+esc(a['filters'])+'</p>'+table([a['metrics']])+charts+'<h2>Insights</h2>'+table(a['insights'])+'<h2>Action plan</h2>'+table(a['actions'])+'<h2>Model evaluation (full dataset)</h2><pre>'+esc(json.dumps(ev,indent=2))+'</pre><h2>Assumptions and limitations</h2><p>'+esc(a['limitations'])+'</p><pre>'+esc(json.dumps(a['criteria'],indent=2))+'</pre></html>'
@app.get('/api/projects/{pid}/download/{kind}')
def download(pid:str,kind:str,machine:str|None=None,category:str|None=None,start:str|None=None,end:str|None=None):
 p=folder(pid)
 if kind=='cleaned': return FileResponse(p/'cleaned.csv',filename='cleaned.csv',media_type='text/csv')
 if kind=='evaluation': return Response(json.dumps(read_evaluation(p)),media_type='application/json',headers={'Content-Disposition':'attachment; filename=evaluation.json'})
 if kind=='pipeline':
  if read_evaluation(p).get('status')!='complete': raise HTTPException(409,'No compatible trained pipeline')
  return FileResponse(p/'pipeline.joblib',filename='pipeline.joblib')
 if kind=='dictionary': return Response(json.dumps({'columns':{c:('hours' if 'Hours' in c else 'count' if c=='Failure_Count' else 'production units (unspecified)' if c=='Production_Volume' else 'identifier/category/date; see setup') for c in dataset(p).columns},'setup':meta(p)['setup'],'missing_values':'Outcomes never imputed; predictors imputed on training only.','unavailable':analysis(p)['comparison'].get('reason')},indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename=data-dictionary.json'})
 a=analysis(p,machine,category,start,end)
 if kind=='priorities': return Response(pd.DataFrame(a['machines']).to_csv(index=False),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=machine-priorities.csv'})
 if kind=='report': return HTMLResponse(report(a,read_evaluation(p)),headers={'Content-Disposition':'attachment; filename=maintenance-report.html'})
 raise HTTPException(404,'Unknown download')
