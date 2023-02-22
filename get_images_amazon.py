import pandas as pd

from tqdm import trange


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

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
keys = itemid_itemimgurl.keys()
for i in trange(len(keys)):
    id = keys[i]
    url_list = itemid_itemimgurl[id]
    counter = 1
    for url in url_list:
        file_name = f'datasets/amazon_imgs/{id}_{counter}.jpg'
        f = open(file_name, 'wb')
        f.write(requests.get(url,headers=headers).content)
        f.close()
        counter += 1

print('Image download finished')
