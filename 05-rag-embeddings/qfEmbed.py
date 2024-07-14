import json
import requests
import os
import chromadb
from chromadb.config import Settings
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
# from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import ChatPromptTemplate
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain.chat_models import QianfanChatEndpoint
from langchain_community.chat_models import QianfanChatEndpoint

# from langchain.embeddings import QianfanEmbeddingsEndpoint
# from langchain_community.embeddings import QianfanEmbeddingsEndpoint
# from qianfan_embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import DocArrayInMemorySearch
# import langchain 
# from qianfan  import QianfanEmbeddingsEndpoint


# 通过鉴权接口获取 access token


def get_access_token11():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token" 
    params = {
        "grant_type": "client_credentials",
        "client_id": os.getenv('ERNIE_CLIENT_ID'),
        "client_secret": os.getenv('ERNIE_CLIENT_SECRET')
    }

    return str(requests.post(url, params=params).json().get("access_token"))

#================

API_KEY = "ZCUQzLB9ku8QkZEZpU8vIEVF"
SECRET_KEY = "vSwKDeCMe6CPQbCr3bWLrqniKNLbe7bG"



# def get_embeddings_bge(prompts):
#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/bge_large_en?access_token=" + get_access_token()
#     print("--------"+str(prompts))
#     payload = json.dumps({
#         "input": prompts
#     })
#     headers = {
#         'Content-Type': 'application/json'
#     }
#     response = requests.request("POST", url, headers=headers, data=payload)
#     # print(response.text)
#     data1 = response.json()
#     data2 = data1.get("data", [])  # 使用get方法获取"data"键的值，如果键不存在则返回空列表
#     if data2:
#         return [x["embedding"] for x in data2]
#     else:
#         print("No 'data' key found in the response.")
#         return []
 
    

def get_access_token():
    _ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 QIANFAN_API_KEY
    ERNIE_CLIENT_ID=os.environ.get("ERNIE_CLIENT_ID")
    ERNIE_CLIENT_SECRET=os.environ.get("ERNIE_CLIENT_SECRET")
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id="+ERNIE_CLIENT_ID+"&client_secret="+ERNIE_CLIENT_SECRET
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

 
#===============


# 调用文心4.0对话接口
# def get_completion_ernie(prompt):

#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
#     payload = json.dumps({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     })

#     headers = {'Content-Type': 'application/json'}

#     response = requests.request(
#         "POST", url, headers=headers, data=payload).json()

#     return response["result"]


#===============================



#===============================
# class QianfanVectorDBConnector:
#     def getQFRetriever(self, embedding_fn):
#             # 使用千帆chatModel
#             # model = QianfanChatEndpoint()
#             chat = QianfanChatEndpoint(
#                 # streaming=True,
#                 QIANFAN_SK="vSwKDeCMe6CPQbCr3bWLrqniKNLbe7bG",
#                 # model="qianfan_chinese_llama_2_7b"
#             )
#             # 使用千帆 embedding bge_large_en 模块
#             embeddings_model = QianfanEmbeddingsEndpoint(model="bge_large_en", endpoint="bge_large_en")

#             # DocArrayInMemorySearch 向量化embedding_fn.len个内容，生成一个数据库对象
#             vectorstore = DocArrayInMemorySearch.from_texts(
#                 embedding_fn,
#                 embedding=embeddings_model,
#             )
#             # 得到向量数据库的一个【检索器】
#             retriever = vectorstore.as_retriever()
    
#             # 构建提示词
#             template = """Answer the question based only on the following context:
#             {context}

#             Question: {question}
#             """
#             prompt = ChatPromptTemplate.from_template(template)

#             # OutputParser， 解析大模型的输出
#             output_parser = StrOutputParser()

#             # 生成一个 RunnableParallel 对象
#             # 这个对象接收一个用户输入（字符串），然后调用retriever，在向量数据库中查询用户输入的内容，
#             # 最后生成一个字典{"context": 'xxxxx', "question": '用户输入问题'}
#             # 这个字典再传给 prompt
#             setup_and_retrieval = RunnableParallel(
#                 {"context": retriever, "question": RunnablePassthrough()}
#             )
#             pre_chain = setup_and_retrieval
#             res_one = pre_chain.invoke("where did harrison work?")
#             print(res_one)

#             # 完整的chain 调用： 用户输入--查询数据库--问题+文档 生成提示词---大模型处理--输出解析
#             chain = setup_and_retrieval | prompt | chat | output_parser

#             res = chain.invoke("how many parameters does llama 2 have?") 
#             print(res)
#             pass

#             # doc = retriever.get_relevant_documents("what do bears like?")
#             # print(doc)
#             # pass


# class MyVectorDBConnector:
#     def __init__(self, collection_name, embedding_fn):
#         chroma_client = chromadb.Client(Settings(allow_reset=True))

#         # 为了演示，实际不需要每次 reset()
#         # chroma_client.reset()

#         # 创建一个 collection
#         self.collection = chroma_client.get_or_create_collection(
#             name=collection_name)
#         self.embedding_fn = embedding_fn
#         # print("__init__ embedding_fn="+str(embedding_fn))
#     def add_documents(self, documents):
#         '''向 collection 中添加文档与向量'''
#         # print("add_documents 's documents="+str(documents))
#         self.collection.add(
#             embeddings=self.embedding_fn(documents),  # 每个文档的向量
#             documents=documents,  # 文档的原文
#             ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
#         )
#         # print("add_documents 's documents="+str(documents))

#     def search(self, query, top_n):
#         '''检索向量数据库'''
#         results = self.collection.query(
#             query_embeddings=self.embedding_fn([query]),
#             n_results=top_n
#         )
#         return results
#============================
class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 为了演示，实际不需要每次 reset()
        chroma_client.reset()

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results
def build_prompt(prompt_template, **kwargs):
    '''将 Prompt 模板赋值'''
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)

# class RAG_Bot:
#     def __init__(self, vector_db, llm_api, n_results=2):
#         self.vector_db = vector_db
#         self.llm_api = llm_api
#         self.n_results = n_results

#     def chat(self, user_query):
#         # 1. 检索
#         search_results = self.vector_db.search(user_query, self.n_results)

#         # 2. 构建 Prompt
#         prompt = build_prompt(
#             prompt_template, context=search_results['documents'][0], query=user_query)

#         # 3. 调用 LLM
#         response = self.llm_api(prompt)
#         return response

#=============================
def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''从 PDF 文件中（按指定页码）提取文字'''
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    # print("extract_text_from_pdf==>"+str(paragraphs))
    return paragraphs


 # 调用文心4.0对话接口
def get_completion_ernie(prompt):

    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    headers = {'Content-Type': 'application/json'}

    response = requests.request(
        "POST", url, headers=headers, data=payload).json()

    return response["result"]


#百度千帆大模型API调用
def open_QFModel(prompt):
    
    #================
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/qianfan_chinese_llama_2_7b?access_token=" + get_access_token()
    
    payload = json.dumps({
         "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        # Get the response content as a dictionary
        response_data = response.json()
        
        # Access the specific data you need from the response_data
        result = response_data.get("result", "")
        
        return result
    else:
        print("Error: API request failed with status code", response.status_code)
        return None
     
    # print(response.text)
    #================
    # 加载环境变量
 
def get_embeddings_bge(prompts):
    # url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/bge_large_en?access_token=" + get_access_token()
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/bge_large_zh?access_token=" + get_access_token()
    payload = json.dumps({
        "input": prompts
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request(
        "POST", url, headers=headers, data=payload).json()
    data = response["data"]
    # print("get_embeddings_bge=>"+str(data))
    return [x["embedding"] for x in data]
# # 调用文心4.0对话接口
# def get_completion_ernie(prompt):

#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
#     payload = json.dumps({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     })

#     headers = {'Content-Type': 'application/json'}

#     response = requests.request(
#         "POST", url, headers=headers, data=payload).json()

#     return response["result"]
class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)

        # 2. 构建 Prompt
        prompt = build_prompt(
            prompt_template, context=search_results['documents'][0], query=user_query)

        # 3. 调用 LLM
        response = self.llm_api(prompt)
        return response
#=================================================
if __name__ == "__main__": 
    # paragraphs = extract_text_from_pdf(".\\05-rag-embeddings\\PDF\\llama2.pdf", min_line_length=10)
    # 为了演示方便，我们只取两页（第一章）
    paragraphs = extract_text_from_pdf(
        # ".\\05-rag-embeddings\\PDF\\llama2.pdf",
        ".\\05-rag-embeddings\\PDF\\06地球毁灭记-五次生物大灭绝7.pdf",
        # page_numbers=[2, 3],
        page_numbers=[16,17],
        min_line_length=10
    )
    # shortened_prompts = [prompt[:16] for prompt in paragraphs]  # 截断长度超过16的部分
    # # # 检查输入数据的长度，如果超过16个字符则截断
    # if len(shortened_prompts) > 16:
    #     input_data = shortened_prompts[:16]  # 截断输入数据至16个字符
    # get_embeddings_bge(input_data)
    #  # # 创建一个向量数据库对象
    new_vector_db = MyVectorDBConnector(
        "demo_ernie_0631",
        embedding_fn=get_embeddings_bge
    )
    # # 向向量数据库中添加文档
    new_vector_db.add_documents(paragraphs)
    # user_query = "how many parameters does llama 2 have?"
    user_query="第三纪是什么的旧称?"
    resultEmb=new_vector_db.search(user_query,5)
    # print("------->"+str(resultEmb))
    
    prompt_template = """
    你是一个问答机器人。
    你的任务是根据下述给定的已知信息回答用户问题。

    已知信息:
    {context}

    用户问：
    {query}

    如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
    请不要输出已知信息中不包含的信息或答案。
    请用中文回答用户问题。
    """
  
    # 创建一个RAG机器人
    new_bot = RAG_Bot(
        new_vector_db,
        llm_api=get_completion_ernie
    )
    chatRes=new_bot.chat(user_query)
    print("------>"+chatRes)
 
    
   
    

 
