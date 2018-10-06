import pandas as pd
import sys

columns = ['session_id', 'item_sku_id', 'user_log_acct', 'operator', 'user_site_cy_name',
           'user_site_province_name', 'user_site_city_name', 'stm_rt', 'page_ts', 'item_name', 'comment_content']

assert len(sys.argv) > 1 
source_file = sys.argv[1]

print('load {}'.format(source_file))
data = pd.read_csv(source_file, names=columns,
                   sep='\t', header=None, skiprows=1)


print('groupby ing')
gb = data.groupby('session_id')
sessions = [gb.get_group(x) for x in gb.groups]

src = []
tgt = []
side = 'test'
for session in sessions:
    session.sort_values('page_ts').to_csv('test.csv', sep='\t',columns=columns,header=False, index=False)
    with open('test.csv', 'r')as f:
        lines = f.readlines()
    sessions_strs = [_l.strip('\n')  for _l in lines]
    # sessions_strs = session.sort_values('page_ts').to_string(columns=columns,header=False, index=False).strip('\n').split('\n')
    if len(sessions_strs)>2:
        src_strs = '||'.join(sessions_strs[:-1])
        tgt_strs = sessions_strs[-1]
        assert(len(tgt_strs.split('\t'))==11)
        src.append(src_strs)
        tgt.append(tgt_strs)

with open('src-{}.txt'.format(side), 'w') as f:
    f.write('\n'.join(src))
    print('save file to src-{}.txt'.format(side))

with open('tgt-{}.txt'.format(side), 'w') as f:
    f.write('\n'.join(tgt))
    print('save file to tgt-{}.txt'.format(side))
    