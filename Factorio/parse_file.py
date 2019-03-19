import numpy as np
import scipy
import os
import re

dir_recipe = r'C:\Program Files (x86)\Steam\steamapps\common\Factorio\data\base\prototypes\recipe'
recipe_files = [os.path.join(dir_recipe, x) for x in os.listdir(dir_recipe)]

string_max = 40
i_file = recipe_files[0]

for i_file in recipe_files:
    with open(i_file, 'r') as f:
        A = f.read()

    for i in A.split('name = '):
        # print('-----', i)
        temp2 = re.sub('\s+', ' ', i)
        temp3 = re.match('.*ingredients =(.*)result.*', temp2)
        if temp3:
            name = re.findall('^.*\n', i)[0]
            space = string_max - len(name)

            match_group = temp3[1]
            print(name.strip(), space * ' ', '-----', match_group)



