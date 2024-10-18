import requests
import json
import pandas as pd
import numpy as np
import os
if __name__ == '__main__':
    centers = np.array([
        [152.40941867408, -27.5041962360992],
        [144.674135858748, -37.3189581380417],
        [116.742389794393, -32.0344032523365],
        [148.909899974392, -34.7496041613316],
        [145.268581263941, -17.8664544609665],
        [131.054736470588, -17.4131120588235],
        [146.84893820442, -41.850354640884],
        [138.885830898502, -34.6915680366057],
        [151.327661106613, -31.7759610391363],
        [149.368995594406, -23.0809256293706]
    ])
    centers = centers[:, [1, 0]]

    BASE_URL = 'https://power.larc.nasa.gov/api/temporal/monthly/point'
    start = '2000'
    end = '2022'
    community = 're'
    parameters = ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'ALLSKY_KT', 'WS50M']
    parameter_string = ','.join(parameters)

    latitude, longitude, cluster, parameter_name, parameter_value, datetime = [], [], [], [], [], []

    for i, (lat, lon) in enumerate(centers):
        response = requests.get(BASE_URL, params={
            'latitude': str(lat),
            'longitude': str(lon),
            'start': start,
            'end': end,
            'community': community,
            'parameters': parameter_string,
            'format': 'json'
        }).json()

        parameters = response['properties']['parameter']
        for param_name, values in parameters.items():
            for year_month_str, value in values.items():
                if year_month_str[4:6] != '13':
                    longitude.append(response['geometry']['coordinates'][0])
                    latitude.append(response['geometry']['coordinates'][1])
                    cluster.append(i)
                    parameter_name.append(param_name)
                    parameter_value.append(value)
                    datetime.append(year_month_str)

    df = pd.DataFrame({
        'latitude': latitude,
        'longitude': longitude,
        'parameter': parameter_name,
        'value': parameter_value,
        'cluster': cluster,
        'datetime': datetime
    })

    # Save the collected data
    output_path = '/opt/ml/processing/output/collected_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data collected and saved to {output_path}")