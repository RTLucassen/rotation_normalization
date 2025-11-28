import pandas as pd
import random


def int_to_roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4, 1
    ]
    syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV", "I"
    ]
    roman = ""
    for i in range(len(val)):
        count = num // val[i]
        roman += syms[i] * count
        num -= val[i] * count
    return roman


def conversion(num):
    return num.replace('1', 'I').replace('2', 'II').replace('3', 'III').replace('4', 'IV').replace('5', 'V').replace('6', 'VI').replace('7', 'VII')


stain_conversions = {
    'MelanAF': 'MELANAF',
    'P16': 'P16',
    'PRAME': 'PRAME',
    'MELAN_AF': 'MELANAF',
    'Ki-67': 'KI67',
    'Melan A': 'MELANA',
    'Mel HMB45': 'MELANAHMB45',
    'KI67': 'KI67',
    'SOX-10 AF': 'SOX10AF',
    'P16 AF': 'P16AF',
    'SOX10_AF': 'SOX10AF',
    'ROS1 AF': 'ROS1AF',
    'BAP-1': 'BAP1',
    'NTRK IHC': 'NTRK',
    'ALK long': 'ALK',
    'PRAME_AF': 'PRAMEAF',
    'NTRK_IHC': 'NTRK',
    'MEL_HMB45': 'MELANAHMB45',
    'ROS1': 'ROS1',
    'BRAF_V600E_AF': 'BRAFAF',
    'ALK_LONG': 'ALK',
    'PRAME AF': 'PRAMEAF',
    'SOX-10': 'SOX10',
    'MEL HMB45 AF': 'MELANAHMB45AF',
    'S100': 'S100',
    'p21': 'P21',
    'MELAN_A': 'MELANA',
    'BRAF_V600E': 'BRAF',
    'BRAF V600E': 'BRAF',
    'SOX10': 'SOX10',
    'BRAF V600E AF': 'BRAFAF',
    'BAP1': 'BAP1',
    'b-Caten': 'BCAT',
    'B_CATEN': 'BCAT',
}

# define number of cases per stain
N = 175

if __name__ == '__main__':

    # define empty list and dictionary to keep track of selected cases
    specimen_nrs = []
    stain_dict = {}

    # collect which stains are available for all cases
    df = pd.read_json('results.json')
    for i, row in df.iterrows():
        slides = row['result']['slides']
        for slide in slides:
            if slide['staining'] in stain_conversions:
                if stain_conversions[slide['staining']] in stain_dict:
                    stain_dict[stain_conversions[slide['staining']]].append((row['pa_number'], row['specimen']))
                else:
                    stain_dict[stain_conversions[slide['staining']]] = [(row['pa_number'], row['specimen'])]

    stain_dict = {k: list(set(v)) for k, v in stain_dict.items()}
    counts = [(len(v),k) for k,v in stain_dict.items()]
    counts = sorted(counts)

    # try random seeds to find a selection with enough unique cases for each stain
    for s in range(100):
        random.seed(s)
        print(s)

        selected_cases = []
        selected = {}
        for n, stain in counts:
            selected[stain] = []
            tries = 0
            while len(selected[stain]) < min(n, N):
                i = random.randint(0, n-1)
                if stain_dict[stain][i] not in selected_cases:
                    selected[stain].append(stain_dict[stain][i])
                    selected_cases.append(stain_dict[stain][i])
                else:
                    tries += 1
                if tries > 10000:
                    break
            if tries > 10000:
                break
        if tries <= 10000:
            break
        for k in selected:
            print(k, len(selected[k]))

    # collect the corresponding paths for the selected cases 
    # and save the information in a json file
    data = {'pa_number': [], 'specimen': [], 'IHC_stain': [], 'HE_path': [], 'IHC_path': []}
    for stain in selected:
        names = [k for k in stain_conversions if stain == stain_conversions[k]]

        matches = []
        for case in selected[stain]:
            df_selection = df[(df['pa_number'] == case[0]) & (df['specimen'] == case[1])]
            if len(df_selection) == 1:
                slides = df_selection['result'].iloc[0]['slides']

                IHC_slides = []
                for slide in slides:
                    if slide['staining'] in names and case[1] == conversion(slide['specimen_nr']):
                        IHC_slides.append(slide)
                
                if len(IHC_slides):
                    for IHC_slide in IHC_slides:
                        corresponding_slides = []
                        for slide in slides:
                            if (('he' in slide['staining'].lower())
                                and (IHC_slide['block'] == slide['block'])
                                and (conversion(IHC_slide['specimen_nr']) == conversion(slide['specimen_nr']))):
                                corresponding_slides.append(slide)
                        if len(corresponding_slides):
                            HE_slide = corresponding_slides[random.randint(0, len(corresponding_slides)-1)]

                            data['pa_number'].append(IHC_slide['pa_number'])
                            data['specimen'].append(conversion(IHC_slide['specimen_nr']))
                            data['IHC_stain'].append(stain)
                            data['HE_path'].append([f"{HE_slide['scan'][-1]['base_dir']}/{name}".replace(r'/data/st1ap-picostr01/pacs',r'T:/dla_pacsarchief') for name in HE_slide['scan'][-1]['files']['SLIDE']])
                            data['IHC_path'].append([f"{IHC_slide['scan'][-1]['base_dir']}/{name}".replace(r'/data/st1ap-picostr01/pacs',r'T:/dla_pacsarchief') for name in IHC_slide['scan'][-1]['files']['SLIDE']])
    
    df = pd.DataFrame.from_dict(data)
    df.to_json('paths.json')
    print(df)