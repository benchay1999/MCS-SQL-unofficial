import os, json
from openai import OpenAI
from utils import open_file, openai_inference, get_db_schema, shuffle_db_schema, filter_db_schema, shuffle_db_schema_column_too, final_filtered_db_schema, search_examples, get_db_description_spider
import pandas as pd
import random
import copy
import re
import sqlite3
import time
from func_timeout import func_set_timeout, FunctionTimedOut, func_timeout
from collections import Counter, defaultdict

class SchemaLinking:
    def __init__(
        self, 
        client:OpenAI, 
        model_name:str="gpt-3.5-turbo"
    ):
        self.client = client
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0


    def table_linking(
            self, 
            db_name:str, 
            question:str,
            db_subname:str="",
            p_t:int=3,
            n:int=20,
            temperature:float=1.0
    )->None:
        """
        This is a table_linking method that does the following:
        
        1. Shuffle the orders of the tables in db_schema `p_t` times to 
        generate `p_t` different prompts.

        2. Then, generate `n` completions for each prompt.
        
        3. Then, we take the union of the tables from the completions 
        to get the final table_linking result.

        Args:
            db_name (str): Name of the database.
            question (str): Question of concern.
            p_t (int): Number of shuffling of the tables in db_schema.
            n (int): Number of completions for each prompt
        """
        if db_name in ["advising", "atis", "ehrsql"]:
            db_schema = get_db_schema(db_name)
        elif db_name in ["spider"]:
            db_schema, self.schema, self.primary, self.foreign = get_db_schema(db_name, db_subname=db_subname)

        self.question = question
        table_list = []

        for i in range(p_t):
            db_schema_shuffled = shuffle_db_schema(db_schema)
            table_linking_prompt = open_file("./prompts/singleDB/table_linking_prompt.txt")\
                .replace("<<<DBSCHEMA>>>", db_schema_shuffled)\
                .replace("<<<QUESTION>>>", question)

            completion = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=[
                {"role": "system", "content" : "Your response should be in a json format."},
                {"role": "user", "content": table_linking_prompt}
                ],
                temperature=temperature,
                n=n
            )
            for i in range(n):
                result = completion.choices[i].message.content
                self.input_tokens += completion.usage.prompt_tokens
                self.output_tokens += completion.usage.completion_tokens
                try:
                    result_json = json.loads(result.replace("\n",""))
                    table_list.extend(result_json["tables"])
                except:
                    pass
        table_list = list(set(table_list))
        self.table_linking_result = []

        # below validates correct table names and filter only the correct ones
        table_column_part = db_schema.split("\n#\n")[0]
        table_column_part_list = table_column_part.split("\n")

        valid_table_name_list = [re.search(r'# (.*?) \(', text).group(1).replace("#","").replace(" ","") for text in table_column_part_list]
        self.table_linking_result = [x for x in table_list if x in valid_table_name_list]

    def column_linking(
            self,
            db_name:str,
            question:str,
            db_subname:str="",
            p_c:int=3,
            n:int=20,
            temperature:float=1.0
    )->None:
        if db_name in ["advising", "atis", "ehrsql"]:
            db_schema = get_db_schema(db_name)
        elif db_name in ["spider"]:
            db_schema, self.schema, self.primary, self.foreign = get_db_schema(db_name, db_subname=db_subname)

        filtered_db_schema = filter_db_schema(db_schema, self.table_linking_result)
        column_list = []
        for i in range(p_c):
            db_schema_shuffled = shuffle_db_schema_column_too(filtered_db_schema)
            column_linking_prompt = open_file("./prompts/singleDB/column_linking_prompt.txt")\
                .replace("<<<DBSCHEMA>>>", db_schema_shuffled)\
                .replace("<<<QUESTION>>>", question)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=[
                {"role": "system", "content" : "Your response should be in a json format."},
                {"role": "user", "content": column_linking_prompt}
                ],
                temperature=temperature,
                n=n
            )
            for i in range(n):
                result = completion.choices[i].message.content
                self.input_tokens += completion.usage.prompt_tokens
                self.output_tokens += completion.usage.completion_tokens
                try:
                    result_json = json.loads(result.replace("\n",""))
                    column_list.extend(result_json["columns"])
                except:
                    pass
        column_list = list(set(column_list))
        #self.column_linking_result = column_list
        self.final_db_schema = final_filtered_db_schema(db_schema, self.table_linking_result, column_list) 
        # below validates correct table_column names and filter only the correct ones
        table_column_part = db_schema.split("\n#\n")[0]
        table_column_part_list = table_column_part.split("\n")
        valid_table_column_name_list = []
        for table_column_part in table_column_part_list:
            table_name = re.search(r'# (.*?) \(', table_column_part).group(1).replace("#", "").replace(" ","")#.strip()
            column_names_str = re.search(r'\( (.*?) \)', table_column_part).group(1).replace("(","").replace(")", "").replace(" ", "")
            column_names_list = column_names_str.split(",")
            valid_table_column_name_list.extend([f"{table_name}.{column_name}" for column_name in column_names_list])
        self.column_linking_result = [x for x in column_list if x in valid_table_column_name_list]
    
    def get_db_description(self, db_name:str, db_subname:str=""):
        """
        Get DB description from `self.final_db_schema`
        """
        db_description_text = ""
        if db_name in ["advising", "atis", "ehrsql"]:
            db_description_text = open_file(f"./prompts/singleDB/{db_name}_db_description.txt")
        elif db_name in ["spider"]:
            db_description_text = open_file(f"./prompts/crossDB/spider/spider_{db_subname}.txt")
        table_description_list = db_description_text.split("\n\n")

        valid_description_list = []
        for idx, table_description in enumerate(table_description_list):
            valid_description = ""
            table_part = ""
            table_part_added_flag = False
            table_name = re.search(r'\[([^\]]+)\]', table_description.split("\n")[0]).group(1).replace("[","").replace("]","")
            if table_name in self.table_linking_result:
                table_part = table_description.split("\n")[0] + "\n"
            for _, column_description in enumerate(table_description.split("\n")[1:]):
                try:
                    column_name = re.search(r'- (.*?) \(', column_description).group(1).replace("-","").replace("(", "").strip()
                except:
                    import pdb; pdb.set_trace()

                table_column_name = f"{table_name}.{column_name}"
                if table_column_name in self.column_linking_result:
                    if not table_part_added_flag:
                        valid_description += table_part
                        table_part_added_flag = True
                    valid_description += column_description + "\n"
            if valid_description:
                valid_description_list.append(valid_description)

        self.final_db_description = "\n".join(valid_description_list).strip()
        
        return self.final_db_description


    def get_db_samples(self, db_name, db_subname=""):
        self.db_sample_list = []
        con = None
        if db_name in ["advising", "atis", "ehrsql"]:
            if db_name in ["advising", "atis"]:
                con = sqlite3.connect(f"./TrustSQL/dataset/{db_name}/{db_name}.sqlite")
            else:
                con = sqlite3.connect(f"./TrustSQL/dataset/{db_name}/mimic_iv.sqlite")
        elif db_name in ["spider"]:
            con = sqlite3.connect(f"./TrustSQL/dataset/{db_name}/database/{db_subname}/{db_subname}.sqlite")
        for table_name in self.table_linking_result:
            column_name_list = [x.split(".")[1] for x in self.column_linking_result if x.split(".")[0]==table_name]
            if not column_name_list:
                continue
            sql = f"""SELECT {", ".join(column_name_list)} FROM '{table_name}' LIMIT 3;"""
            tab = pd.read_sql_query(sql, con)
            db_sample_list = [re.sub(r'^\d', '', line).strip() for line in str(tab).split("\n")[1:]]
            db_sample_list = [re.sub(r' {2,}', ', ', text) for text in db_sample_list]
            
            db_sample_prompt = f"# [{table_name}]\n" + ", ".join(column_name_list) + "\n" + "\n".join(db_sample_list).strip()
            self.db_sample_list.append(db_sample_prompt)
        return self.db_sample_list
    
class SQLGeneration:
    def __init__(
        self,
        client:OpenAI,
        model_name:str="gpt-3.5-turbo"
    ):
        self.client = client
        self.model_name = model_name
        self.example_dict = {
            "masked_example_list" : [],
            "unmasked_example_list" : [],
            "masked_prompt" : "",
            "unmasked_prompt" : "",
            "hybrid_prompt_list" : [] # length: k - 2
        }
        self.input_tokens = 0
        self.output_tokens = 0
        
    def question_masking(
            self,
            db_name:str,
            question:str,
            db_subname:str=""
    )->str:
        """
        A method to generate masked question.

        Args:
            example_dict (dict[List]): A dictionary containing lists of `db_name_list`, `question_list`, and `masked_question_list`/
            db_name (str): Name of the database.
            question (str): Question of concern.

        """
        #examples = ""
        question_masking_prompt = ""
        if db_name in ["advising", "atis", "ehrsql"]:
            if db_name == "advising":
                from prompts.singleDB.question_masking_fewshot import advising_question_masking_fewshot as examples
            elif db_name == "atis":
                from prompts.singleDB.question_masking_fewshot import atis_question_masking_fewshot as examples
            else:
                from prompts.singleDB.question_masking_fewshot import ehrsql_question_masking_fewshot as examples
            question_masking_prompt = open_file("./prompts/singleDB/question_masking_prompt.txt")\
                .replace("<<<EXAMPLES>>>", examples)\
                .replace("<<<DBSCHEMA>>>", get_db_schema(db_name))\
                .replace("<<<QUESTION>>>", question)
        elif db_name in ["spider"]:
            from prompts.crossDB.question_masking_fewshot import spider_question_masking_fewshot as examples
            question_masking_prompt = open_file("./prompts/crossDB/question_masking_prompt.txt")\
                .replace("<<<EXAMPLES>>>", examples)\
                .replace("<<<DBSCHEMA>>>", get_db_schema(db_name=db_name, db_subname=db_subname)[0])\
                .replace("<<<QUESTION>>>", question)

        completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Generate only the answer."},
                    {"role": "user", "content": question_masking_prompt}
                ],
                temperature=0.2,
                n=1
            )
        result = completion.choices[0].message.content
        self.input_tokens += completion.usage.prompt_tokens
        self.output_tokens += completion.usage.completion_tokens
        question_masking_result = {
            "original" : question,
            "masked" : result
        }
        return question_masking_result
    
    def retrieve_examples(
        self,
        db_name:str,
        question:str,
        db_subname:str="",
        p_q:int=5
    )->None:
        """
        Generates fewshot prompt to inject to SQL Generation prompt
        """
        if p_q < 2:
            raise ValueError("`p_q` needs to be higher than 2.")
        if db_name in ["advising", "atis", "ehrsql", "spider"]:
            data_path = f"./vectorDB/{db_name}_train.pkl"
        
        data = pd.read_pickle(data_path)

        # unmasked question
        ret_instances = search_examples(data, question, self.client, n=20) # dataframe with "question", "sql", "question_embedding", "similarities"

        prompt = "<examples>\n"
        for idx, (question_, sql) in enumerate(zip(ret_instances["question"].values, ret_instances["sql"].values)):
            question_prompt = f"# Question: {question_}\n"
            sql_prompt = f"# Gold SQL: {sql}\n"
            self.example_dict["unmasked_example_list"].append(question_prompt + sql_prompt)
            prompt = prompt + question_prompt + sql_prompt
        prompt += "</examples>"

        self.example_dict["unmasked_prompt"] = prompt
        
        # masked question
        # this is "야매". To do it properly, I think we should also mask all the questions in the training set.
        # However, due to budget constraints, we only mask the input question, not the questions in the training set.
        masked_question = self.question_masking(db_name=db_name, question=question, db_subname=db_subname)["masked"]
        ret_instances = search_examples(data, masked_question, self.client, n=20)
        prompt = "<examples>\n"
        for idx, (question_, sql) in enumerate(zip(ret_instances["question"].values, ret_instances["sql"].values)):
            question_prompt = f"# Question: {question_}\n"
            sql_prompt = f"# Gold SQL: {sql}\n"
            self.example_dict["masked_example_list"].append(question_prompt + sql_prompt)
            end = "\n" if idx < len(ret_instances["question"].values)-1 else ""
            prompt = prompt + question_prompt + sql_prompt + end
        prompt += "</examples>"

        self.example_dict["masked_prompt"] = prompt

        # hybrid
        for i in range(p_q-2):
            unmasked_number = random.randint(1,19)
            masked_number = 20 - unmasked_number
            unmasked_idx_list = random.sample(range(len(self.example_dict["unmasked_example_list"])), unmasked_number)
            masked_idx_list = random.sample(range(len(self.example_dict["masked_example_list"])), masked_number)
            unmasked_examples = [self.example_dict["unmasked_example_list"][idx] for idx in unmasked_idx_list]
            masked_examples = [self.example_dict["masked_example_list"][idx] for idx in masked_idx_list]

            examples = copy.deepcopy(unmasked_examples)
            examples.extend(masked_examples)
            random.shuffle(examples)

            prompt = "<examples>\n"
            for idx, example_instance in enumerate(examples):
                end = "\n" if idx < len(examples)-1 else ""
                prompt += example_instance + end
            prompt += "</examples>"

            self.example_dict["hybrid_prompt_list"].append(prompt)
        return self.example_dict
    def generate_sql(
        self,
        question:str,
        schema_linker:SchemaLinking,
        temperature:float=1.0,
        n:int=20
    )->None:
        db_schema = schema_linker.final_db_schema
        db_description = schema_linker.final_db_description
        db_sample_list = schema_linker.db_sample_list
        db_samples = "\n\n".join(db_sample_list)
        example_dict = self.example_dict

        sql_list = []
        # unmasked
        sql_generation_prompt = open_file("./prompts/singleDB/sql_generation_prompt.txt")\
            .replace("<<<EXAMPLES>>>", example_dict["unmasked_prompt"])\
            .replace("<<<DBSCHEMA>>>", db_schema)\
            .replace("<<<DBDESCRIPTION>>>", db_description)\
            .replace("<<<DBSAMPLES>>>", db_samples)\
            .replace("<<<QUESTION>>>", question)
        completion = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=[
                {"role": "system", "content" : "Your response should be in a json format."},
                {"role": "user", "content": sql_generation_prompt}
                ],
                temperature=temperature,
                n=n
        )
        for i in range(n):
            result = completion.choices[i].message.content
            self.input_tokens += completion.usage.prompt_tokens
            self.output_tokens += completion.usage.completion_tokens
            try:
                result_json = json.loads(result.replace("\n",""))
                sql_list.append(result_json["sql"])
            except:
                pass
        # masked
        sql_generation_prompt = open_file("./prompts/singleDB/sql_generation_prompt.txt")\
            .replace("<<<EXAMPLES>>>", example_dict["masked_prompt"])\
            .replace("<<<DBSCHEMA>>>", db_schema)\
            .replace("<<<DBDESCRIPTION>>>", db_description)\
            .replace("<<<DBSAMPLES>>>", db_samples)\
            .replace("<<<QUESTION>>>", question)
        completion = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=[
                {"role": "system", "content" : "Your response should be in a json format."},
                {"role": "user", "content": sql_generation_prompt}
                ],
                temperature=temperature,
                n=n
        )
        for i in range(n):
            result = completion.choices[i].message.content
            self.input_tokens += completion.usage.prompt_tokens
            self.output_tokens += completion.usage.completion_tokens
            try:
                result_json = json.loads(result.replace("\n",""))
                sql_list.append(result_json["sql"])
            except:
                pass
        for prompt_instance in example_dict["hybrid_prompt_list"]:
            sql_generation_prompt = open_file("./prompts/singleDB/sql_generation_prompt.txt")\
            .replace("<<<EXAMPLES>>>", prompt_instance)\
            .replace("<<<DBSCHEMA>>>", db_schema)\
            .replace("<<<DBDESCRIPTION>>>", db_description)\
            .replace("<<<DBSAMPLES>>>", db_samples)\
            .replace("<<<QUESTION>>>", question)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=[
                {"role": "system", "content" : "Your response should be in a json format."},
                {"role": "user", "content": sql_generation_prompt}
                ],
                temperature=temperature,
                n=n
            )
            for i in range(n):
                result = completion.choices[i].message.content
                self.input_tokens += completion.usage.prompt_tokens
                self.output_tokens += completion.usage.completion_tokens
                try:
                    result_json = json.loads(result.replace("\n",""))
                    sql_list.append(result_json["sql"])
                except:
                    pass
        self.sql_list = sql_list
    def candidate_filtering(self, db_name, db_subname:str=""):
        """
        Filter based on:
            1. Running time: should be faster than 180 seconds.
            2. Confidence scores: ratio of other sql candidates yielding the same execution results. Threshold: 0.2
        """
        con = None
        cursor = None
        self.sql_valid_list = []
        if db_name in ["advising", "atis", "ehrsql"]:
            con = sqlite3.connect(f"./TrustSQL/dataset/{db_name}/{db_name}.sqlite",check_same_thread=False)
            cursor = con.cursor()
        elif db_name in ["spider"]:
            con = sqlite3.connect(f"./TrustSQL/dataset/{db_name}/database/{db_subname}/{db_subname}.sqlite",check_same_thread=False)
            cursor = con.cursor()
        for sql in self.sql_list:
            start_time = time.time()
            results = ""
            try:
                func_timeout(180, cursor.execute, args=(sql,))
                results = cursor.fetchall()
                execution_time = time.time() - start_time
                self.sql_valid_list.append((sql, results, execution_time))
            except:
                results = "<<<ERROR OR TIMEOUT>>>"
                execution_time = float("inf")
        result_dict = defaultdict(list)
        result_counts = Counter(str(result) for _, result, _ in self.sql_valid_list)
        total_count = sum(result_counts.values())
        for sql, result, execution_time in self.sql_valid_list:
            count = result_counts[str(result)]
            result_dict[str(result)].append({"sql": sql, "confidence_score":count/total_count, "execution_time":execution_time})

        result_dict = dict(result_dict)
        
        #self.result_dict = result_dict

        min_time_dict = {}
        for result, instance in result_dict.items():
            min_instance = min(instance, key=lambda x: x["execution_time"])
            min_time_dict[str(result)] = {"sql": min_instance["sql"], "confidence_score": min_instance["confidence_score"]}
        min_time_dict = min_time_dict
        
        filtered_candidates = {}
        for result, instance in min_time_dict.items():
            if instance["confidence_score"] >= 0.2:
                filtered_candidates[instance["sql"]] = {"result" : result, "confidence_score": instance["confidence_score"]}
        self.filtered_candidates = filtered_candidates
    def multiple_choices_selection(
        self,
        db_name:str,
        question:str,
        schema_linker:SchemaLinking,
        db_subname:str="",
        temperature:float=1.0,
        n:int=20
    ):
        db_schema = schema_linker.final_db_schema
        db_description = schema_linker.final_db_description
        db_sample_list = schema_linker.db_sample_list
        db_samples = "\n\n".join(db_sample_list)
        example_dict = self.example_dict


        # candidates should be listed in the order of confidence scores
        candidates = [key for key, instance in sorted(self.filtered_candidates.items(), key=lambda x: x[1]["confidence_score"], reverse=True)]
        candidate_prompt = ""
        for idx, candidate in enumerate(candidates):
            end = "\n" if idx < len(candidates)-1 else ""
            candidate = candidate.replace("\n", " ")
            candidate_prompt += f"{idx+1}. {candidate}{end}"
        # we use masked prompt for multiple choices selection because the paper claimed (on crossDB though) that retrieval from masked question performed better.
        sql_selection_prompt = open_file("./prompts/singleDB/sql_selection_prompt.txt")\
            .replace("<<<EXAMPLES>>>", example_dict["masked_prompt"])\
            .replace("<<<DBSCHEMA>>>", db_schema)\
            .replace("<<<DBDESCRIPTION>>>", db_description)\
            .replace("<<<DBSAMPLES>>>", db_samples)\
            .replace("<<<QUESTION>>>", question)\
            .replace("<<<CANDIDATES>>>", candidate_prompt)
        
        completion = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=[
                {"role": "system", "content" : "Your response should be in a json format."},
                {"role": "user", "content": sql_selection_prompt}
                ],
                temperature=temperature,
                n=n
        )
        
        selection_result_counter = Counter()
        for i in range(n):
            result = completion.choices[i].message.content
            self.input_tokens += completion.usage.prompt_tokens
            self.output_tokens += completion.usage.completion_tokens
            try:
                result_json = json.loads(result.replace("\n",""))
                selection_result_counter[result_json["sql"]] += 1
            except:
                pass
        # select the result with the highest count
        self.final_sql = selection_result_counter.most_common(1)[0][0]
        # run the final SQL query
        con = None
        cursor = None
        if db_name in ["advising", "atis", "ehrsql"]:
            con = sqlite3.connect(f"./TrustSQL/dataset/{db_name}/{db_name}.sqlite",check_same_thread=False)     
        elif db_name in ["spider"]:
            con = sqlite3.connect(f"./TrustSQL/dataset/{db_name}/database/{db_subname}/{db_subname}.sqlite")
        cursor = con.cursor()
        try:
            cursor.execute(self.final_sql)
            self.final_sql_result = cursor.fetchall()
        except:
            self.final_sql_result = "<<<ERROR OR TIMEOUT>>>"
        

# TODO: implement for crossDB(SPIDER) also.
# as of now: implemented loading full DB schema when the db_name is given.
# TODO 1: implement db description too
# TODO 2: implement db samples too (is it feasible?)
# TODO 3: make prompts for question masking
# TODO 4: make prompts for others too (pay heed in retrieving examples)
