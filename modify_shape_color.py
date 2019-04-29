import os, sys
import json


def get_new_json(filepath, key):
    key_ = key.split(".")
    key_length = len(key_)
    colorarrow = [255, 0, 0, 128]
    colorinhibit = [0, 255, 0, 128]
    colorgene = [0, 0, 0, 128]
    with open(filepath, 'r') as f:
        json_data = json.load(f)
        i = 0
        a = json_data['shapes']
        b=a[0]['points']
        length=len(a)
        while i<len(a):
            b=a[i]['label']
            if a[i]['label'].find('gene:')==0:
                a[i][key] = colorgene
            elif a[i]['label'].find('activate:')==0:
                a[i][key] = colorarrow
            elif a[i]['label'].find('inhibit:')==0:
                a[i][key]=colorinhibit
            i=i+1
        # while i < key_length:
        #     if i + 1 == key_length:
        #         a[key_[i]] = value
        #         i = i + 1
        #     else:
        #         a = a[key_[i]]
        #         i = i + 1
    f.close()
    return json_data


def rewrite_json_file(filepath, json_data):
    with open(filepath, 'w') as f:
        json.dump(json_data, f,indent=2)
    f.close()


if __name__ == '__main__':
    # key = sys.argv[1]
    # value = int(sys.argv[2])
    # json_path = sys.argv[3]
    jsonpath=r'C:\Users\LSC-110\Desktop\Fei'
    for jsonfile in os.listdir(jsonpath):
        if os.path.splitext(jsonfile)[-1] == '.json':
            m_json_data = get_new_json(os.path.join(jsonpath,jsonfile), 'line_color')
            rewrite_json_file(os.path.join(jsonpath,jsonfile), m_json_data)
