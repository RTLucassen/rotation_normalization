import json

import requests
import pandas as pd
from tqdm import tqdm


# define path to spreadsheet
spreadsheet_path = r""

# define path to output file and the url of slideslinger
output_path = "output_case.json"
url = ""

if __name__ == '__main__':

    # load spreadsheet
    df = pd.read_excel(spreadsheet_path)

    pa_numbers = []
    specimens = []
    for _, row in df.iterrows():
        items = row['pa_number_specimen'].split('+')
        for item in items:
            pa_number, specimen = item.split('_')
            pa_numbers.append(pa_number)
            specimens.append(specimen)

    # collect the request results and add them to the dataframe before saving
    results = {'pa_number': [], 'specimen': [], 'result': []}
    for pa_number, specimen in tqdm(zip(pa_numbers, specimens)):
        query = {'pa_number': pa_number}
        response = requests.get(f'{url}/case/', data=json.dumps(query)).json()

        if response is not None:
            results['pa_number'].append(pa_number)
            results['specimen'].append(specimen)
            results['result'].append(response)

    df_results = pd.DataFrame.from_dict(results)
    df_results.to_json('results.json', index=False)