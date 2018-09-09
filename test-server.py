from collections import Counter
import argparse
import requests
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Path to csv')
parser.add_argument(
    'server',
    help='The full address to the server. If localhost, will be something '
         'like http://127.0.0.1:5000/. If on heroku, will look like '
         'https://<your-app-name>.herokuapp.com. Please note that you will '
         'NOT include the "predict" portion of the URL, we will do that!'
)
parser.add_argument(
    '-n', '--n_samples', help='Number of samples', type=int, default=10
)
parser.add_argument(
    '-r', '--random_state', help='Seed to pick samples', default=42
)
args = parser.parse_args()

X_test = pd.read_csv(
    args.dataset
).sample(
    n=args.n_samples, random_state=args.random_state
)

results = []
for i, obs in X_test.iterrows():
    payload = {
        'id': i,
        'observation': obs.to_dict()
    }
    results.append(requests.post(
        '{}predict'.format(args.server),
        json=payload
    ))

print('status code results:', Counter([
    r.status_code
    for r in results
]))
