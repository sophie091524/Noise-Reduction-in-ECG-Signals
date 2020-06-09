import os
import numpy as np
from glob import iglob
from scipy.io import loadmat, savemat
from os.path import join, basename, splitext

TYPE = ['clean', '-2.5db', '0db', '2.5db', '5db', '7.5db']
PERSON = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']

def main(dirname):
    #os.mkdir('dataset')
    for m in TYPE:
        if not os.path.exists('dataset/'+ m):
            os.mkdir('dataset/'+ m)
        node = 1024
        val = np.zeros((1, node))
        val_lab = []
        for p in PERSON:
            label = []
            path = join(dirname, m, p )
            data = np.zeros((1, node))
            for f in iglob(path + '/*mat'):
                filename = join(path, f)
                mat = loadmat(filename)
                name = splitext(basename(f))[0]
                label.append(name)
                sig = mat['temp']
                #print(sig.shape)
                #sig = np.reshape(sig, (5, node))
                data = np.concatenate([data, sig], axis=0)
            val = np.concatenate([val, data[1:]], axis=0)
            val_lab.append(np.asarray(label[:]))
            #print(test_lab)
        #print(val.shape)
        np.save('dataset/' + m + '/val', val[1:])
        np.save('dataset/' + m + '/val_name', val_lab)
        print(val.shape)
        #print(val_lab)

if __name__ == '__main__':
    main('/mnt/intern/user_sophie/short_ecg_0_1/val')





