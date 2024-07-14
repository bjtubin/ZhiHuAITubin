# from typing import Dict, Any
import requests   
from bs4 import BeautifulSoup  # type: ignore
import os

def fetch_webpage(url):
    try:
        # 发送HTTP GET请求，并指定接受的编码为utf-8
        response = requests.get(url)
        
        # 检查响应状态码
        response.raise_for_status()
        
        # 明确指定网页编码为utf-8
        response.encoding = 'utf-8'
        
        return response.text
    
    except requests.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return None

def save_to_file(data, filename):
    try:
        # 构建文件完整路径
        filepath = os.path.join(os.getcwd(), filename)
        
        # 写入数据到文件，使用utf-8编码
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(data)
        
        print(f"Data has been saved to {filepath}")
    
    except IOError as e:
        print(f"Error saving data to file: {e}")

def main():
    # url = 'https://www.gov.cn/zhengce/index.htm'
    # url = "https://www.gov.cn/search/zhengce/?t=zhengce&q=数据要素&timetype=&mintime=&maxtime=&sort=score&sortType=&searchfield=&pcodeJiguan=&childtype=&subchildtype=&tsbq=&pubtimeyear=&puborg=&pcodeYear=&pcodeNum=&filetype=&p=&n=&inpro=&sug_t="
    # webpage_content = fetch_webpage(url)
    
    query = "数据要素"
    url = f"https://www.gov.cn/search/zhengce/?t=zhengce&q=&timetype=&mintime=&maxtime=&sort=score&sortType=&searchfield=&pcodeJiguan=&childtype=&subchildtype=&tsbq=&pubtimeyear=&puborg=&pcodeYear=&pcodeNum=&filetype=&p=&n=&inpro=&sug_t="
    webpage_content = fetch_webpage(url)
    
    if webpage_content:
        # 使用BeautifulSoup解析网页内容，并指定解析器使用的编码为utf-8
        soup = BeautifulSoup(webpage_content, 'html.parser')#, from_encoding='utf-8')
        
        # 获取所有的段落元素
        paragraphs = soup.find_all('p')
        
        # 将段落元素转换为字符串
        data = '\n'.join(p.get_text() for p in paragraphs)
        
        # 保存数据到文件
        save_to_file(data, 'gov_data.txt')

if __name__ == '__main__':
    main()