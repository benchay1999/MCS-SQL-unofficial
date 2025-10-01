
from mcs_sql import SchemaLinking
import os
from openai import OpenAI
model_name = "gpt-3.5-turbo"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

schema_linker = SchemaLinking(client=client, model_name=model_name)


question = "What are the methods for ingesting oxymetazoline?"
schema_linker.table_linking(db_name="ehrsql", question=question)
print("****************Table linking result****************")
print(schema_linker.table_linking_result)

schema_linker.column_linking(db_name="ehrsql", question=question)
print("****************Column linking result****************")
print(schema_linker.column_linking_result)
print("****************Filtered DB Schema****************")
print(schema_linker.final_db_schema)
print("****************Filtered DB Samples****************")
db_samples = schema_linker.get_db_samples("ehrsql")
for sample in db_samples:
    print(sample)
print("****************FilteredDB Description****************")
print(schema_linker.get_db_description(db_name="ehrsql"))
print("****************Schema Linking Usage****************")
print(f"total input tokens: {schema_linker.input_tokens}\t total output tokens: {schema_linker.output_tokens}")

from mcs_sql import SQLGeneration
sg = SQLGeneration(client=client, model_name=model_name)
sg.retrieve_examples(db_name="ehrsql", question=question)

sg.generate_sql(question=question, schema_linker=schema_linker)
sg.candidate_filtering("ehrsql")
sg.multiple_choices_selection(db_name = "ehrsql", question=question, schema_linker=schema_linker)
print("****************Result SQL****************")
print(sg.final_sql)
print("****************SQL Generation Execution Result****************")
print(sg.final_sql_result)
print("****************SQL Generation Usage****************")
print(f"total input tokens: {sg.input_tokens}\t total output tokens: {sg.output_tokens}")
print()
print("****************Total Tokens****************")
print(f"input: {schema_linker.input_tokens+sg.input_tokens}")
print(f"output: {schema_linker.output_tokens+sg.output_tokens}")
print("*******************************************")
import pdb; pdb.set_trace()
