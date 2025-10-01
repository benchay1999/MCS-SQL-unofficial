import sqlite3
import pandas as pd
import random
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
import re



def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def get_db_schema(db_name:str, db_subname:str="")->str:
    """
    Each line of the db_schema should start with "# "
    """
    db_schema = ""
    if db_name in ["ehrsql", "atis", "advising"]:
        if db_name == "ehrsql":
            from prompts.singleDB.db_schema import ehrsql_db_schema as db_schema
        elif db_name == "atis":
            from prompts.singleDB.db_schema import atis_db_schema as db_schema
        else:
            from prompts.singleDB.db_schema import advising_db_schema as db_schema
        return db_schema
    if db_name in ["spider"]:
        schema, primary, foreign = load_schema("./TrustSQL/dataset/spider/tables.json")
        db_schema = create_schema_prompt_type1(db_subname, schema, primary, foreign, db_path="./TrustSQL/dataset/spider/database")
        return db_schema, schema, primary, foreign

def shuffle_db_schema(db_schema:str)->str:
    """
    Shuffle the order of the tables in the db_schema
    """
    table_column = db_schema.split("\n#\n")[0]
    foreign_keys = db_schema.split("\n#\n")[1]
    table_column_before = table_column.split("\n")
    random.shuffle(table_column_before)

    table_column_after = "\n".join(table_column_before)
    db_schema = table_column_after + "\n#\n" + "\n".join(foreign_keys)
    return db_schema

def filter_db_schema(db_schema:str, table_list:list)->str:
    """
    Filter the db_schema to only include the tables in table_list
    """
    table_column = db_schema.split("\n#\n")[0]
    foreign_keys = db_schema.split("\n#\n")[1:]
    table_column_before = table_column.split("\n")
    table_column_after = []
    for line in table_column_before:
        table_name = line.split(" (")[0].replace("# ","")
        if table_name in table_list:
            table_column_after.append(line)
    table_column_after = "\n".join(table_column_after)

    foreign_keys_before = foreign_keys
    foreign_keys_after = []
    for line in foreign_keys_before:
        if "=" in line:
            table_name_1 = line.split("=")[0].split(".")[0].replace("# ","").strip()
            table_name_2 = line.split("=")[1].split(".")[0].strip()
            if table_name_1 in table_list and table_name_2 in table_list:
                foreign_keys_after.append(line)
    if foreign_keys_after:
        db_schema = table_column_after + "\n#\n" + "\n".join(foreign_keys_after)
    else:
        db_schema = table_column_after
    return db_schema

def final_filtered_db_schema(db_schema:str, table_list:list, column_list:list)->str:
    """
    Final filtered db_schema
    table_list is a list of `table_names`
    column_list is a list of `table_names.column names`

    From the db_schmea, we filter out the tables and columns that are not in the table_list and column_list
    """
    table_column = db_schema.split("\n#\n")[0]
    foreign_keys = db_schema.split("\n#\n")[1:]
    table_column_before = table_column.split("\n")
    table_column_after = []
    for line in table_column_before:
        table_name = re.search(r'# (.*?) \(', line).group(1).replace("#", "").replace(" ","")#.strip()
        column_names_str = re.search(r'\( (.*?) \)', line).group(1).replace("(","").replace(")", "").replace(" ", "")
        column_names_list = column_names_str.split(",")
        columns_after = []
        for column in column_names_list:
            if f"{table_name}.{column}" in column_list:
                columns_after.append(column)
        if columns_after:
            table_column_after.append(f"# {table_name} ( {', '.join(columns_after)} )")
    table_column_after = "\n".join(table_column_after)

    foreign_keys_before = "".join(foreign_keys).split("\n")
    foreign_keys_after = []
    for line in foreign_keys_before:
        if "=" in line:
            table_column_1 = line.split("=")[0].replace("# ","").strip()
            table_column_2 = line.split("=")[1].strip()
            if table_column_1 in column_list and table_column_2 in column_list:
                foreign_keys_after.append(line)
    if foreign_keys_after:
        final_db_schema = table_column_after + "\n#\n" + "\n".join(foreign_keys_after)
    else:
        final_db_schema = table_column_after
    return final_db_schema

def shuffle_db_schema_column_too(db_schema:str)->str:
    """
    Shuffle the order of the tables and columns in the db_schema
    """
    table_column = db_schema.split("\n#\n")[0]
    foreign_keys = db_schema.split("\n#\n")[1:]
    table_column_before = table_column.split("\n")
    table_column_after = []
    for line in table_column_before:
        table_name = line.split(" (")[0].replace("# ","")
        columns = line.split(" (")[1].split(" )")[0].split(", ")
        random.shuffle(columns)
        table_column_after.append(f"# {table_name} ( {', '.join(columns)} )")
    random.shuffle(table_column_after)
    table_column_after = "\n".join(table_column_after)

    db_schema = table_column_after + "\n#\n" + "\n".join(foreign_keys)
    return db_schema
def get_embedding(text, client, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_examples(df, product_description, client, n=5, model="text-embedding-ada-002"):
    embedding = get_embedding(product_description, client, model=model)
    #df.question_embedding = df.question_embedding.apply(lambda x: np.array(ast.literal_eval(x)))
    embeddings_matrix = np.vstack(np.array(df.question_embedding.values))

    similarities = cosine_similarity(embeddings_matrix, np.array(embedding)[None, :])
    df['similarities'] = similarities.flatten()
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

def openai_inference(model_name, message, client, temperature=0.2, top_p=0.1, json_mode=False, seed=42, n=1):
    if not json_mode:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
            {"role": "user", "content": message}
            ],
            temperature=temperature,
            top_p = top_p,
            seed=seed,
            n=n
        )
    else:
        completion = client.chat.completions.create(
            model=model_name,
            response_format={ "type": "json_object" },
            messages=[
            {"role": "system", "content" : "Your response should be in a json format."},
            {"role": "user", "content": message}
            ],
            temperature=temperature,
            top_p = top_p,
            seed=seed,
            n=n
        )
    return completion.choices[0].message.content

def load_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index > -1:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    db_schema = pd.DataFrame(schema, columns=['Database name', 'Table Name', 'Field Name', 'Type'])
    primary_key = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    foreign_key = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return db_schema, primary_key, foreign_key

def create_schema_prompt_type1(db_id, schema, primary, foreign, db_path):
    schema = schema[schema['Database name']==db_id]
    primary = primary[primary['Database name']==db_id]
    foreign = foreign[foreign['Database name']==db_id]
    prompt = ''
    foreign_key_prompt = ''
    tab_names = []
    for tab_name in schema['Table Name'].values:
        if tab_name not in tab_names:
            tab_names.append(tab_name)
    for tab_name in tab_names:
        cols = schema['Field Name'][schema['Table Name']==tab_name].values
        types = schema['Type'][schema['Table Name']==tab_name].values
        prompt += f'# {tab_name} ( '
        for idx, (col, type_) in enumerate(zip(cols, types)):
            #col, type_ = col, type_           
            if idx < len(cols) - 1:
                comma = ', '
            elif idx == 0:
                comma = ''
            elif idx == len(cols) - 1:
                comma = ' )'
            prompt += f'{col}{comma}'
        prompt += "\n"
        primary_cols = primary['Primary Key'][primary['Table Name']==tab_name].values
        if len(primary_cols)>0:
            for col in primary_cols:
                col = col
        foreign_cols = foreign[['Second Table Name', 'First Table Foreign Key', 'Second Table Foreign Key']][foreign['First Table Name']==tab_name].values
        if len( foreign_cols)>0:
            for tab2, col1, col2 in foreign_cols:
                col1, tab2, col2 = col1, tab2, col2
                foreign_key_prompt += f"# {tab_name}.{col1} = {tab2}.{col2}\n"

    result = prompt.strip() + "\n#\n" + foreign_key_prompt.strip()
    return result


def get_db_description_spider(schema, table_column_list):
    
    prompt = ''
    tab_names = []
    for tab_name in schema['Table Name'].values:
        if tab_name not in tab_names:
            tab_names.append(tab_name)

    for tab_name in tab_names:
        cols = schema['Field Name'][schema['Table Name']==tab_name].values
        types = schema['Type'][schema['Table Name']==tab_name].values
        table_name_added_flag = False
        for idx, (col, type_) in enumerate(zip(cols, types)):
            col, type_ = col, type_            
            if idx < len(cols) - 1:
                comma = ', '
            elif idx == 0:
                comma = ''
            elif idx == len(cols) - 1:
                comma = ' )'
            
            if f"{tab_name}.{col}" in table_column_list:
                if not table_name_added_flag:
                    prompt += f'# [{tab_name}]\n'
                    table_name_added_flag = True
                prompt += f'- {col} ({type_}): <<<DESCRIPTION>>>\n'
        if table_name_added_flag:
            prompt += "\n"

    prompt = prompt.strip()
    

    return prompt.strip()

def get_db_description_crossdb(db_name:str, table_column_list:list, db_subname:str="")->str:
    """
    Get the description of the database
    """
    db_description = ""
    
    if db_name in ["spider"]:
        pass
    return db_description

def get_full_spider_description(db_id, schema, primary, foreign, db_path):
    schema = schema[schema['Database name']==db_id]
    primary = primary[primary['Database name']==db_id]
    foreign = foreign[foreign['Database name']==db_id]
    con = sqlite3.connect(f'{db_path}/{db_id}/{db_id}.sqlite')
    prompt = ''
    tab_names = []
    for tab_name in schema['Table Name'].values:
        if tab_name not in tab_names:
            tab_names.append(tab_name)

    for tab_name in tab_names:
        sql = f'SELECT DISTINCT * FROM {tab_name} LIMIT 5'
        tab = pd.read_sql_query(sql, con)
        value_examples_prompt = ""
        cols = schema['Field Name'][schema['Table Name']==tab_name].values
        types = schema['Type'][schema['Table Name']==tab_name].values
        table_name_added_flag = False
        for idx, (col, type_) in enumerate(zip(cols, types)):
            col, type_ = col, type_            
            if idx < len(cols) - 1:
                comma = ', '
            elif idx == 0:
                comma = ''
            elif idx == len(cols) - 1:
                comma = ' )'
            
            if not table_name_added_flag:
                prompt += f'# [{tab_name}]\n'
                table_name_added_flag = True
            value_examples_prompt += f"Value examples for {tab_name}.{col}: {', '.join([str(l) for l in tab[col]])}\n"

            prompt += f'- {col} ({type_}): <<<DESCRIPTION>>>\n'
        if table_name_added_flag:
            prompt += "\n"

    prompt = prompt.strip()
    prompt += "\n\n" + value_examples_prompt.strip()
    
    return prompt.strip()