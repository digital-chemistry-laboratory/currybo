import json
import numpy as np

def is_smiles(label):
    # a smiles is always a string
    if not isinstance(label, str):
        return False
    # array
    if len(label.split(' ')) > 1:
        return False
    # scalar
    try:
        float(label)
        return False
    except: # smiles
        return True

def convert_to_array(label):
    if isinstance(label, str):
        arr = np.fromstring(label, sep=' ')
    else:
        arr = np.array([label])
    return arr if arr.size else None

def find_closest_label(point, labels_numpy, rtol=1e-5):
    arr = convert_to_array(point)
    if arr is None:
        return None
    for key, value in labels_numpy.items():
        if np.allclose(arr, value, rtol=rtol):
            return key
    return None

def print_output(output, dataset, cli_args):
    labels_numpy = {key: convert_to_array(key) for key in dataset.labels.keys() if not is_smiles(key)}

    for i in cli_args['conditions']:
        point = output['estimated_current_optimum']['point'][i['name']]
        if point in dataset.labels:
            output['estimated_current_optimum']['point'][i['name']] = f"{dataset.labels[point]};{point}"
        elif is_smiles(point):
            output['estimated_current_optimum']['point'][i['name']] = point
        elif find_closest_label(point, labels_numpy) is not None:
            closest_label = find_closest_label(point, labels_numpy)
            output['estimated_current_optimum']['point'][i['name']] = f"{dataset.labels[closest_label]};{closest_label}"

    for i in [*cli_args['conditions'], *cli_args['substrates']]:
        for j, k in enumerate(output['next_points']):
            point = k['point'][i['name']]
            if point in dataset.labels:
                output['next_points'][j]['point'][i['name']] = f"{dataset.labels[point]};{point}"
            elif is_smiles(point):
                output['next_points'][j]['point'][i['name']] = point
            elif find_closest_label(point, labels_numpy) is not None:
                closest_label = find_closest_label(point, labels_numpy)
                output['next_points'][j]['point'][i['name']] = f"{dataset.labels[closest_label]};{closest_label}"

    # if cli_args['json']:
    j = json.dumps(output, indent=2) 
    print(j)
    # else:
    #     print("CurryBO suggests the following measurements:")
    #     print("")

    #     for o in output:
    #         for k in o.keys():
    #             print(f"- {k}: {o[k]}")
    #         print("")
