import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json


with open('relevant_priors_public.json', 'r') as f:
    data = json.load(f)

truth = {(item['case_id'], item['study_id']): item['is_relevant_to_current'] for item in data['truth']}

cases = data.get('cases', [])

rows = []
for case in cases:
    case_id = case.get('case_id', '')
    current = case.get('current_study', {})
    current_desc = current.get('study_description', '')
    current_date = current.get('study_date', '')
    curr_dt = None
    try:
        curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
    except Exception:
        pass
    for prev in case.get('prior_studies', []):
        study_id = prev.get('study_id', '')
        prev_desc = prev.get('study_description', '')
        prev_date = prev.get('study_date', '')
        prev_dt = None
        try:
            prev_dt = datetime.strptime(prev_date, "%Y-%m-%d")
        except Exception:
            pass
        same_desc = int(prev_desc == current_desc)
        days_diff = (curr_dt - prev_dt).days if curr_dt and prev_dt else 9999
        keyword_overlap = len(set(current_desc.split()) & set(prev_desc.split()))
        label = truth.get((case_id, study_id), False)
        rows.append({
            'case_id': case_id,
            'study_id': study_id,
            'same_desc': same_desc,
            'days_diff': days_diff,
            'keyword_overlap': keyword_overlap,
            'label': int(label)
        })

features = pd.DataFrame(rows)
X = features[['same_desc', 'days_diff', 'keyword_overlap']]
y = features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

joblib.dump(clf, 'relevance_model.joblib')
print('Model saved as relevance_model.joblib')
