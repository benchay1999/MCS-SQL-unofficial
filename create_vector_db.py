from mcs_sql import SchemaLinking
import os, json
from openai import OpenAI
import pandas as pd
import numpy as np
model_name = "gpt-3.5-turbo"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

db_name = "spider"
data_path = f"/home/wschay/TrustSQL/dataset/{db_name}/{db_name}_train.json"
with open(data_path, "r") as f:
    data = json.load(f)
question_list = [d["question"] for d in data]
sql_list = [d["query"] for d in data]
print(len(question_list))
print(len(sql_list)) 
question_embedding = [get_embedding(q) for q in question_list]
df = pd.DataFrame({"question": question_list, "sql": sql_list, "question_embedding": question_embedding})
df.to_pickle(f"/home/wschay/MCS-SQL/vectorDB/{db_name}_train.pkl")