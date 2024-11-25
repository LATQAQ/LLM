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

# alpaca 格式
# 这里使用instruction和 output
# [
#   {
#     "instruction": "user instruction (required)",
#     "input": "user input (optional)",
#     "output": "model response (required)",
#     "system": "system prompt (optional)",
#     "history": [
#       ["user instruction in the first round (optional)", "model response in the first round (optional)"],
#       ["user instruction in the second round (optional)", "model response in the second round (optional)"]
#     ]
#   }
# ]

# 最终格式
# {
#     "instruction": "user instruction (required)",
#     "output": "model response (required)",
#     "system": "system prompt (optional)",
# }

# 数据预处理  生成alpaca格式
# path = 'Data/DnD5E_text'
# output_path = 'Data/json'

# sys_prompt = "你将扮演一位经验丰富的《龙与地下城》（DND 5E）地下城主（DM）。你的任务是为玩家创造一个身临其境的奇幻冒险体验。你应清晰、热情且引人入胜地描述场景、非玩家角色（NPC）和世界事件，同时确保规则严谨且公正。你还需要灵活应对玩家行动，设计富有挑战性的遭遇，推动故事的发展，并保持游戏节奏的流畅性。在交互中：你是世界的叙述者，为玩家描绘环境、提供感官细节，以及表达NPC的行为和情绪。根据玩家行动解释规则，进行骰子判定，并确保游戏规则和逻辑一致。根据剧情需要，灵活编排事件和遭遇，鼓励创造性解决问题。如果玩家不熟悉规则，耐心提供相关解释，简化复杂问题的理解。请以引人入胜的方式回答，保持对话趣味性，并推动冒险故事向前发展。现在，续写以下文本："


# count = 0
# for root, dirs, files in os.walk(path):
#     count += 1
#     for file in files:
#         with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
#             text = f.read()

#             text = text.replace('\n', ' ')  # 去掉换行符
#             # 通过滑动窗口的方式生成instruction和output, 窗口大小为512
#             window_size = 512
#             text_len = len(text)
#             # 最后一个窗口不足512的话，填入空格
#             if text_len % window_size != 0:
#                 text += ' ' * (window_size - text_len % window_size)
#             # 生成instruction和output
#             for i in range(0, text_len, window_size):
#                 instruction = text[i:i + window_size]
#                 output = text[i + window_size:i + 2 * window_size]
#                 # 写入文件
#                 output_file = os.path.join(output_path, root.split('/')[-1], file.split('.')[0] + '\\' + str(i) + '.json')
#                 output_file = output_file.replace('\\DnD5E_text', '')
#                 if not os.path.exists(os.path.dirname(output_file)):
#                     os.makedirs(os.path.dirname(output_file))
#                 with open(output_file, 'w', encoding='utf-8') as of:
#                     of.write('{"instruction": "' + instruction + '", "output": "' + output + '", "system": "' + sys_prompt + '"}')

#     if count % 40 == 0:
#         print('已处理{}个文件'.format(count))

output_path = 'Data/my_dataset.json'
path = 'Data/DnD5E_text'

sys_prompt = "你将扮演一位经验丰富的《龙与地下城》（DND 5E）地下城主（DM）。你的任务是为玩家创造一个身临其境的奇幻冒险体验。你应清晰、热情且引人入胜地描述场景、非玩家角色（NPC）和世界事件，同时确保规则严谨且公正。你还需要灵活应对玩家行动，设计富有挑战性的遭遇，推动故事的发展，并保持游戏节奏的流畅性。在交互中：你是世界的叙述者，为玩家描绘环境、提供感官细节，以及表达NPC的行为和情绪。根据玩家行动解释规则，进行骰子判定，并确保游戏规则和逻辑一致。根据剧情需要，灵活编排事件和遭遇，鼓励创造性解决问题。如果玩家不熟悉规则，耐心提供相关解释，简化复杂问题的理解。请以引人入胜的方式回答，保持对话趣味性，并推动冒险故事向前发展。现在，续写以下文本："


count = 0

lst = []

for root, dirs, files in os.walk(path):
    count += 1
    for file in files:
        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
            text = f.read()

            text = text.replace('\n', ' ')  # 去掉换行符
            # 通过滑动窗口的方式生成instruction和output, 窗口大小为512
            window_size = 512
            text_len = len(text)
            # 最后一个窗口不足512的话，填入空格
            if text_len % window_size != 0:
                text += ' ' * (window_size - text_len % window_size)
            # 生成instruction和output
            for i in range(0, text_len, window_size):
                instruction = text[i:i + window_size]
                output = text[i + window_size:i + 2 * window_size]

                lst.append({"instruction": instruction, "output": output, "system": sys_prompt})            

    if count % 40 == 0:
        print('已处理{}个文件'.format(count))

import json
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(lst, f, ensure_ascii=False, indent=4)
print('处理完成')