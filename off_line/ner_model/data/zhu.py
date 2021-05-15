import json
import os

def func1():
    file_data = open('validate.txt', 'r', encoding='utf-8')
    result_data = open('new_validate.txt', 'w', encoding='utf-8')    

    for line in file_data.readlines():
        line = line.strip('\n').strip()
        data = json.loads(line)
        feature = data['text']
        label = data['label']
        if len(feature) != len(label):
            print('ERROR!')

        for i in range(len(feature)):
            fea = feature[i]
            lab = label[i]
            comb = fea + '\t' + lab
            result_data.write(comb + '\n')

        result_data.write('\n')

    file_data.close()
    result_data.close()


if __name__ == '__main__':
    func1()

