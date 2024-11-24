from transformers import AutoTokenizer, AutoModelForCausalLM
import chardet
import os

# 检测是否全为utf-8编码
# path = 'Data/DnD5E_text'
# count = 0
# for root, dirs, files in os.walk(path):
#     count += 1
#     for file in files:
#         with open(os.path.join(root, file), 'rb') as f:
#             content = f.read()
#             result = chardet.detect(content)
#             if result['encoding'] != 'utf-8':
#                 print('------------------------------------------------------------------------')
#                 print(os.path.join(root, file))
#                 print('------------------------------------------------------------------------')
#                 print(result)
#     if count % 40 == 0:
#         print('已检测{}个文件'.format(count))


# 数据预处理
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
path = 'Data/DnD5E_text'
count = 0




