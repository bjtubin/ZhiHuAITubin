# 读取文件
 
import chardet

filepath = 'D:\\GitHub\\ZhiHuAITubin\\docs\\aaa.docx'
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

encoding = detect_encoding(filepath)
print(f"The detected encoding is: {encoding}")

# with open(filepath, 'r', encoding='utf-8') as file:
#     content = file.read()
#     print(content)
    
  
# print(detect_encoding(filepath))

# 写入文件
# with open('path/to/your/newfile.txt', 'w') as file:
#     file.write("Hello, world!")