import json


# 读取JSON格式的字典
def read_json_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# 替换文本文件中的字典键为值，并将结果保存到新文件
def replace_keys_with_values_and_save(input_txt_file_path, output_txt_file_path, json_dict):
    with open(input_txt_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 替换文本中的每个键
    for key, value in json_dict.items():
        text = text.replace(key, value)

    # 将替换后的文本写入新文件
    with open(output_txt_file_path, 'w', encoding='utf-8') as file:
        file.write(text)


# 主函数
def main():
    json_file_path = './CCKS2019/ner_data/tag_file.json'
    input_txt_file_path = './CCKS2019/ner_data/dev.txt'
    output_txt_file_path = './CCKS2019/ner_data/dev_zh.txt'

    # 读取字典
    json_dict = read_json_dict(json_file_path)

    # 替换文本文件中的键，并保存到新文件
    replace_keys_with_values_and_save(input_txt_file_path, output_txt_file_path, json_dict)


# 调用主函数
if __name__ == '__main__':
    main()
