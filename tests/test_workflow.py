import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend import app as m

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    return TestClient(m.app)

def row(**kw):
    r = dict(Machine_ID='M1', Machine_Category='CNC', Maintenance_Date='2025-01-05', Downtime_Hours=2, Production_Volume=20, Failure_Count=1)
    r.update(kw)
    return r

def project(client, rows):
    raw = pd.DataFrame(rows).to_csv(index=False).encode()
    r = client.post('/api/projects', files={'file': ('data.csv', raw, 'text/csv')})
    assert r.status_code == 200, r.text
    return r.json()['id']

def setup(client, pid, **kw):
    r = client.put(f'/api/projects/{pid}/setup', json=m.Setup(**kw).model_dump())
    assert r.status_code == 200, r.text

def test_validation(client):
    for raw in [b'', b'foo\n1']:
        assert client.post('/api/projects', files={'file': ('x.csv', raw)}).status_code == 422
    pid = project(client, [row(), row(), row(Downtime_Hours=None, Maintenance_Date='invalid'), row(Failure_Count=-1, Production_Volume='oops')])
    q = client.get(f'/api/projects/{pid}/quality').json()
    assert q['duplicates_removed'] == 1
    assert q['valid_outcomes']['Downtime_Hours'] == 2
    assert any('invalid dates' in i['issue'] for i in q['issues'])
    assert m.dataset(m.folder(pid)).Downtime_Hours.isna().sum() == 1
    assert not client.get(f'/api/projects/{pid}/analysis').json()['comparison']['available']
    assert client.get('/api/projects/not-an-id').status_code == 404

def test_events_filters_exports(client):
    pid = project(client, [row(Observation_Date=f'2025-01-{i:02}') for i in range(1, 10)])
    setup(client, pid, row_meaning='daily', event_meaning='event_date')
    assert client.get(f'/api/projects/{pid}/analysis').json()['metrics']['events'] == 1
    assert client.get(f'/api/projects/{pid}/analysis?start=2025-01-04&end=2025-01-06').json()['metrics']['records'] == 3
    assert client.get(f'/api/projects/{pid}/analysis?machine=absent').json()['metrics']['records'] == 0
    for kind in ['cleaned', 'priorities', 'report', 'dictionary', 'evaluation']:
        r = client.get(f'/api/projects/{pid}/download/{kind}')
        assert r.status_code == 200, r.text
    r = client.get(f'/api/projects/{pid}/download/report')
    assert '<svg' in r.text and '4. Machine priority overview' in r.text
    assert client.post(f'/api/projects/{pid}/events', files={'file': ('e.csv', b'Machine_ID,Maintenance_Date\nM1,bad')}).status_code == 422

def test_paired_zero_coverage_overlap():
    s = m.Setup(row_meaning='daily', window_days=3, minimum_coverage=1)
    dates = pd.date_range('2025-01-01', periods=15)
    d = pd.DataFrame({'Machine_ID':'M1', 'Observation_Date':dates, 'Downtime_Hours':[0.]*7+[2.]*8, 'Failure_Count':1, 'Production_Volume':10})
    e = pd.DataFrame({'Machine_ID':['M1'], 'Maintenance_Date':[dates[7]]})
    a = m.paired(d,e,s)
    assert a['pairs'][0]['reduction_pct'] is None
    assert a['pairs'][0]['absolute_change'] == 2
    d.loc[6,'Downtime_Hours'] = np.nan
    assert not m.paired(d,e,s)['pairs']
    e = pd.concat([e,pd.DataFrame({'Machine_ID':['M1'], 'Maintenance_Date':[dates[8]]})])
    assert m.paired(d,e,s)['excluded']['overlapping'] == 2

def test_features_exclude_future_and_gaps():
    d = pd.DataFrame([row(Observation_Date=pd.Timestamp(dt), Downtime_Hours=v) for dt,v in [('2025-01-01',1),('2025-01-02',99),('2025-01-05',4)]])
    e = pd.DataFrame({'Machine_ID':['M1'], 'Maintenance_Date':[pd.Timestamp('2025-01-02')]})
    out,cols,mode = m.features(d,e,m.Setup(row_meaning='daily',event_meaning='event_date'))
    assert len(out) == 1 and out.iloc[0]['target'] == 99
    assert out.iloc[0]['previous_Downtime_Hours'] == 1
    assert out.iloc[0]['historical_events'] == 0 and 'Machine_ID' not in cols

def test_forecast_training_prediction_reload_stale(client):
    rows = [row(Machine_ID=f'M{mid}', Observation_Date=str(date.date()), Maintenance_Performed=int(i==8), Maintenance_Date='2025-01-09' if i==8 else '', Downtime_Hours=float((i+mid)%5), Production_Volume=100+i) for mid in range(6) for i,date in enumerate(pd.date_range('2025-01-01',periods=45))]
    pid = project(client,rows)
    setup(client,pid,row_meaning='daily',event_meaning='indicator')
    assert client.post(f'/api/projects/{pid}/train').status_code == 200
    ev = client.get(f'/api/projects/{pid}/evaluation').json()
    assert ev['status'] == 'complete', ev
    assert ev['train_last_target'] < ev['test_first_target']
    values = {c:2 for c in ev['features']}
    values['Machine_Category'] = 'Unseen'
    r = client.post(f'/api/projects/{pid}/predict',json=values)
    assert r.status_code == 200, r.text
    assert 'Unseen machine category' in r.json()['warnings']
    assert m.joblib.load(m.folder(pid)/'pipeline.joblib') is not None
    setup(client,pid,row_meaning='daily',event_meaning='indicator',window_days=20)
    assert client.post(f'/api/projects/{pid}/predict',json=values).status_code == 409

def test_explanatory_group_separation(client):
    pid = project(client,[row(Machine_ID=f'M{mid}',Downtime_Hours=(i+mid)%4,Production_Volume=100+i) for mid in range(8) for i in range(20)])
    setup(client,pid,row_meaning='event')
    client.post(f'/api/projects/{pid}/train')
    ev = client.get(f'/api/projects/{pid}/evaluation').json()
    assert ev['status'] == 'complete', ev
    assert ev['mode'] == 'explanatory'
    assert not set(ev['train_machines']) & set(ev['test_machines'])

def test_explicit_duration_category_and_invalid_failure(client):
    rows = [row(Machine_ID='A', Period_Start='2025-01-01', Period_End='2025-01-01', Downtime_Hours=30),
            row(Machine_ID='B', Period_Start='2025-01-05', Period_End='2025-01-01'),
            row(Machine_ID='C', Failure_Count=1.5),
            row(Machine_ID='D'), row(Machine_ID='D', Machine_Category='Other')]
    pid=project(client,rows)
    d=m.dataset(m.folder(pid))
    assert set(d.Machine_ID)=={'A','C'}
    assert d.loc[d.Machine_ID=='A','Downtime_Hours'].isna().all()
    assert d.loc[d.Machine_ID=='C','Failure_Count'].isna().all()

def test_rule_support_matches_training_tree(client):
    from sklearn.tree import DecisionTreeRegressor
    tree=DecisionTreeRegressor(max_depth=2).fit([[0],[1],[2],[3]],[0,0,2,2])
    rules=m.decision_rules(tree,['prior downtime'])
    assert sum(r['training_support'] for r in rules)==4
    assert all('prior downtime' in r['conditions'] for r in rules)
