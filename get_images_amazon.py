import pandas as pd

from tqdm import trange
import wget

df = pd.read_csv('../../FolkScope/TOTAL_typicality_result.csv')
# df = pd.read_csv('TOTAL_typicality_result.csv')

itemid_itemimgurl = {}
for i in trange(len(df)):
    ida = df['item_a_id'][i]
    idb = df['item_b_id'][i]
    if ida not in itemid_itemimgurl:
        itemid_itemimgurl[ida] = []
    if idb not in itemid_itemimgurl:
        itemid_itemimgurl[idb] = []
    if df['item_a_img1_validity'][i] != 0:
        url = df['item_a_img1_url'][i]
        if url not in itemid_itemimgurl[ida]:
            itemid_itemimgurl[ida].append(url)
    if df['item_a_img2_validity'][i] != 0:
        url = df['item_a_img2_url'][i]
        if url not in itemid_itemimgurl[ida]:
            itemid_itemimgurl[ida].append(url)
    if df['item_a_img3_validity'][i] != 0:
        url = df['item_a_img3_url'][i]
        if url not in itemid_itemimgurl[ida]:
            itemid_itemimgurl[ida].append(url)

    if df['item_b_img1_validity'][i] != 0:
        url = df['item_b_img1_url'][i]
        if url not in itemid_itemimgurl[idb]:
            itemid_itemimgurl[idb].append(url)
    if df['item_b_img2_validity'][i] != 0:
        url = df['item_b_img2_url'][i]
        if url not in itemid_itemimgurl[idb]:
            itemid_itemimgurl[idb].append(url)
    if df['item_b_img3_validity'][i] != 0:
        url = df['item_b_img3_url'][i]
        if url not in itemid_itemimgurl[idb]:
            itemid_itemimgurl[idb].append(url)




import requests
import shutil

for id,url_list in itemid_itemimgurl.items():
    counter = 1
    for url in url_list:
        print(url)
        file_name = wget.download(url)
        quit()
        file_name = f'datasets/amazon_imgs/{id}_{counter}.img'
        f = open(file_name, 'wb')
        f.write(requests.get(url).content)
        f.close()
        counter += 1
    quit()
