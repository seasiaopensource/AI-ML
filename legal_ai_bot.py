import uvicorn
from typing import List, Dict, Any
import glob
import openai
from langchain.llms import OpenAI
from prompt import prompt_template
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, UploadFile, Request,HTTPException
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pydantic import BaseModel
import mysql.connector
import os
import re
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import OpenAIEmbeddings


load_dotenv()

#PineCone details
pinecone_key = os.getenv('pinecone_key')
pinecone_env = os.getenv('pinecone_env')
index_name = os.getenv('index_name')
# nameSpace = "test-namespace"

pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_env,
)

# OpenAI details
os.environ["OPENAI_API_KEY"] = os.getenv("openai_key")  ## client api
openai.api_key = os.getenv("openai_key")




base_dir = os.getcwd()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]                           
)

class vector(BaseModel):
    database_name:str
    id: int

class Highlight(BaseModel):
    word: str | None = None
    database_name:str


class Comparision(BaseModel):
    updated_paragraph: str
    original_paragraph: str

class all_quesans(BaseModel):
    database_name:str
    query: str


class delete(BaseModel):
    database_name:str
    id: int

class ques_ans(BaseModel):
    database_name:str
    query: str
    name_space: str


# Function to store the embeddings into the PineCone DataBase
def insertion(database_name,sub_name, texts):
    namespace_nm = f'{database_name}_{sub_name}'

    if not texts:
        return f"There's no data in the contract"
    else:

        embed = OpenAIEmbeddings()
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        if namespace_nm in stats.namespaces:
            return f'Embeddings already exist'
        else:

            nameSpace = f'{database_name}_All_Pdf'
            Pinecone.from_documents(documents=texts, embedding=embed, index_name=index_name,
                                    namespace=nameSpace)
            Pinecone.from_documents(documents=texts, embedding=embed, index_name=index_name,
                                    namespace=namespace_nm)
        return f'Embeddings Created'


# Function to query and store the PDf Text
@app.post('/vector_stores/')
async def vector_stores(item:vector):
    # items = item.dict()
    items = item.model_dump()
    database_name=items['database_name']
    id = items['id']

    try:
        db_config = {
            'host': os.getenv('db_host'),
            'user': os.getenv('db_user'),
            'password': os.getenv('db_password'),
            'database': database_name,
        }
        connection = mysql.connector.connect(**db_config)
    except mysql.connector.Error as conn_err:
        print(conn_err)
        raise HTTPException(status_code=400, detail=f'Something went wrong')

    try:
        cursor = connection.cursor()
        query_ = (f'''SELECT section_id ,clause, pdf_name,pdf_id,title,
                CASE WHEN row_num = 1 THEN CONCAT(title, ' ', clause) ELSE clause END AS merge_column
                FROM (
                    SELECT section_clauses.section_id,section_clauses.clause,Contracts.name AS pdf_name,Contracts.id AS pdf_id,contract_details.title,
                    ROW_NUMBER() OVER (PARTITION BY section_clauses.section_id ORDER BY section_clauses.clause) AS row_num
                    FROM section_clauses
                     JOIN contract_details ON contract_details.id=section_clauses.section_id
                     JOIN Contracts ON Contracts.id=contract_details.contract_id
                    WHERE Contracts.id = {id}
                    ) AS data; ''')

        cursor.execute(query_)
        result = cursor.fetchall()

        clean_text = []
        for item in result:
            par = str(item[5])
            # Check if i is a string before applying re.sub
            if isinstance(par, str):
                plain_text = re.sub(r'<.*?>', '', par)
                clean_text.append(Document(page_content=plain_text,
                                           metadata={"Pdf_Name": item[2], "Pdf_Id": item[3], "Section_Id": item[0]}))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_text = text_splitter.split_documents(clean_text)
        txt = insertion(database_name, f'{id}', split_text)

    except mysql.connector.Error as err:
        print(err)
        raise HTTPException(status_code=400, detail=f'Something went wrong')

    finally:
        if connection:
            cursor.close()
            connection.close()
    return {'success': True , 'status':200,'message': txt}




# Function to question answer from All pdfs in that Database
@app.post("/question_answer")
async def question_answer(item: all_quesans):
    items = item.dict()
    database_name=items['database_name']
    query = items['query']

    try:
        embed = OpenAIEmbeddings()
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()

        nameSpace = f"{database_name}_All_Pdf"
        if nameSpace in stats.namespaces:
            vectorstore = Pinecone.from_existing_index(index_name='ailegalbot', embedding=embed, text_key="text",
                                                       namespace=nameSpace)


            retriever = vectorstore.as_retriever(search_type="similarity")  # search_kwargs={"k": 13}
            llm = OpenAI(temperature=0.5, model_name='gpt-3.5-turbo')
            # llm = ChatOpenAI(model_name='gpt-3.5-turbo')

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt_template,
            )
            question_answers = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
                verbose=False,
                return_source_documents=True,
                chain_type_kwargs={
                    "verbose": False,
                    "prompt": prompt,
                }
            )
            response = question_answers({"query": query})
            detail_info = response['source_documents']

            unique_pdf_names = set(doc.metadata["Pdf_Name"] for doc in detail_info)

            return {'success': True, 'status': 200, 'result': response["result"], 'Pdf_Name': unique_pdf_names,
                    'query': query}

        else:
            return {'success': False ,'status':400,'message':'No database is there'}

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=400, detail='Something went wrong')



if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port = 8000)
