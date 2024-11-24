import requests
from bs4 import BeautifulSoup
import os
import chardet

# url = 'http://dndlogs.com/5E/indexh.htm'
# webhelp = 'http://dndlogs.com/5E/webhelplefth.htm'

# webhelp_response = requests.get(webhelp)
# webhelp_response.encoding = "utf-8"

# webhelp_soup = BeautifulSoup(webhelp_response.text, 'html.parser')

# # 子页面格式 'http://dndlogs.com/5E/index.htm?page=玩家手册/种族/提夫林.html'
# template =  'http://dndlogs.com/5E/{page}'

# # 从webhelp页面中提取所有子页面的链接
# links = []
# for texts in webhelp_soup.find_all('a'):
#     link = texts.get('href')
#     if link != '#':
#         links.append(link)

# # 下载所有子页面
# count = 0
# for link in links:
#     count += 1
#     response = requests.get(template.format(page=link))
#     charset = chardet.detect(response.content)['encoding']
#     response.encoding = charset
#     print(charset)
    
#     link = link[7:]
#     if not os.path.exists('Data/raw_html_DnD5E/' + link):
#         os.makedirs(os.path.dirname('Data/raw_html_DnD5E/' + link), exist_ok=True)
#     with open('Data/raw_html_DnD5E/' + link, 'w', encoding='utf-8') as f:
#         f.write(response.text)

#     print('Data/raw_html_DnD5E/' + link)

#     if count % 40 == 0:
#         print('已下载{}个文件'.format(count))

path = 'Data/raw_html_DnD5E'

count = 0

for root, dirs, files in os.walk(path):
    
    count += 1

    for file in files:
        if not file.endswith('.html'):
            continue
        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
            # print('------------------------------------------------------------------------')
            # print(os.path.join(root, file))
            # print('------------------------------------------------------------------------')
            content = f.read()
            soup = BeautifulSoup(content, 'html.parser')

            text = soup.get_text()
            cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            file = file.replace('.html', '.txt')
            
            root_ = root.replace('Data/raw_html_DnD5E', 'Data/DnD5E_text/')
            root_ = root_.replace('\\', '/')
            root_ = root_.replace('//', '/')
            root_ = root_+'/'

            if not os.path.exists(root_+file):
                os.makedirs(os.path.dirname(root_+file), exist_ok=True)
            with open(root_+file , 'w', encoding='utf-8') as f:
                f.write(cleaned_text)  
    if count % 40 == 0:
        print('已处理{}个文件'.format(count))