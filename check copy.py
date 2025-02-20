import os

directory = '/home/zezhi/QinDeyun/Pointnet2_DeYun/data/Dataset_DeYun/v2_object_z_rotation'

for filename in sorted(os.listdir(directory)):
    if filename.startswith('label_03') and filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            print(f'Contents of {filename}:')
            for i in range(100):
                line = file.readline()
                if not line:
                    break
                print(line.strip())
            print('---')