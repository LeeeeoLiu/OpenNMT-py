import pandas as pd
import sys

columns = ['session_id', 'item_sku_id', 'user_log_acct', 'operator', 'user_site_cy_name',
           'user_site_province_name', 'user_site_city_name', 'stm_rt', 'page_ts', 'item_name', 'comment_content']

assert len(sys.argv) > 1 
source_file = sys.argv[1]

print('load {}'.format(source_file))
data = pd.read_csv(source_file, names=columns,
                   sep='\t', header=None, skiprows=1)

print(data.info())
assert 1==0
f_columns = ['session_id', 'user_log_acct', 'operator', 'user_site_cy_name',
             'user_site_province_name', 'user_site_city_name', 'item_name', 'comment_content']
for _c in f_columns:
    print('filter \"  \" in column:{}'.format(_c))
    data[_c] = data[_c].map(lambda x: x.replace('  ',''))

print('groupby ing')
gb = data.groupby('session_id')
sessions = [gb.get_group(x) for x in gb.groups]

pos = len(sessions)-5000 if len(sessions) > 30000 else len(sessions) - \
    int(len(sessions)*0.1)
train_sessions = sessions[:pos]
valid_sessions = sessions[pos:]

print('all session:{}'.format(len(sessions)))
print('train:{}'.format(len(train_sessions)))
print('valid:{}'.format(len(valid_sessions)))

for side in ['train', 'valid']:
    _sessions = train_sessions if side == 'train' else valid_sessions
    src = []
    tgt = []
    for session in _sessions:
        sessions_strs = session.sort_values('page_ts').to_string(columns=columns,header=False, index=False).strip('\n').split('\n')
        if len(sessions_strs)>2:
            src_strs = '||'.join(sessions_strs[:-1])
            tgt_strs = sessions_strs[-1]
            src.append(src_strs)
            tgt.append(tgt_strs)

    with open('src-{}.txt'.format(side), 'w') as f:
        f.write('\n'.join(src))
        print('save file to src-{}.txt'.format(side))

    with open('tgt-{}.txt'.format(side), 'w') as f:
        f.write('\n'.join(tgt))
        print('save file to tgt-{}.txt'.format(side))
        

