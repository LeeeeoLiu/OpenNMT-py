import pandas as pd


columns = ['session_id', 'item_sku_id', 'user_log_acct', 'operator', 'user_site_cy_name',
           'user_site_province_name', 'user_site_city_name', 'stm_rt', 'page_ts', 'item_name', 'comment_content']

data = pd.read_csv('test_data/src-train.txt', names=columns,
                   sep='\t', header=None, skiprows=1)

gb = data.groupby('session_id')
sessions = [gb.get_group(x) for x in gb.groups]

pos = len(sessions)-5000 if len(sessions) > 30000 else len(sessions) - \
    int(len(sessions)*0.1)
train_sessions = sessions[:pos]
valid_sessions = sessions[pos:]

print('all session:{}'.format(len(sessions)))
print('train:{}'.format(len(train_sessions)))
print('valid:{}'.format(len(valid_sessions)))

src = []
tgt = []
for session in train_sessions:
    sessions_strs = session.sort_values('page_ts').to_string(columns=columns,header=False, index=False).strip('\n').split('\n')
    if len(sessions_strs)>2:
        src_strs = '||'.join(sessions_strs[:-1])
        tgt_strs = sessions_strs[-1]
        src.append(src_strs)
        tgt.append(tgt_strs)

with open('test_data/src-train-new.txt', 'w') as f:
    f.write('\n'.join(src))

with open('test_data/tgt-train-new.txt', 'w') as f:
    f.write('\n'.join(tgt))

