from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain import LLMChain

import pandas as pd
import ast
import textwrap
import os
import time
from argparse import ArgumentParser
import sys
import glob
import json
import re


import os
import time
from argparse import ArgumentParser
import sys
import json
import math
import requests
from func_timeout import func_set_timeout
import func_timeout


import sqlite3

langchain.verbose = False

TABLE_PARAMETER = "{TABLE_PARAMETER}"
c_tr_index = "{c_tr_index}"
DROP_TABLE_SQL = f"DROP TABLE {TABLE_PARAMETER};"
GET_TABLES_SQL = "SELECT name FROM sqlite_schema WHERE type='table';"
GET_ROW_SQL = f"""SELECT * FROM {TABLE_PARAMETER} WHERE "index" = {c_tr_index};"""
def delete_all_tables(con):
    tables = get_tables(con)
    delete_tables(con, tables)

def get_row(con, c_table, c_index):
    cur = con.cursor()
    sql = GET_ROW_SQL.replace(TABLE_PARAMETER, c_table); sql = sql.replace(c_tr_index, str(c_index))
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    return rows


def get_tables(con):
    cur = con.cursor()
    cur.execute(GET_TABLES_SQL)
    tables = cur.fetchall()
    cur.close()
    return tables


def delete_tables(con, tables):
    cur = con.cursor()
    for table, in tables:
        sql = DROP_TABLE_SQL.replace(TABLE_PARAMETER, table)
        cur.execute(sql)
    cur.close()

@func_set_timeout(40)
def table2text(llm, tableRow, question):
    llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(template_PM3_table2text)
    )

    result = llm_chain.generate([{"tableData":tableRow, "question":question}])
    return result.generations[0][0].text


@func_set_timeout(40)
def tableNtext_qa(llm, tableRow, pt, question):
    llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(template_PM3_tableNtext_qa)
    )

    result = llm_chain.generate([{"tableData":tableRow, "pt":pt, "question":question}])
    return result.generations[0][0].text

@func_set_timeout(40)
def wrapper(func, query):
    return(func(query))



def is_number(s):
    try: 
        float(s)
        return True
    except ValueError:
        pass

    return False


# for benchmarking only, generate half-sturctured data in plain text from single table_row
template_PM3_table2text = """
### System:
You are reading the structured data given in the Context and try to rephrase it in plain text. In each line, the attribute name(header) is on the left of *:*, then corresponding attribute value is on the right.

### Context:
{tableData}

### User:
Each variant/mutation must contain alphabet letters with several digits, don't make up non-existed variants/mutations. 
Limit your answer under 25 words.
Stop the answer by the word *END*.
Please read the above provided structured data in context and just answer the given question in short plain text. Question: {question}'\
### Response:

"""




template_PM3_tableNtext_qa = """
### System:
You are reading the structured data and it's corresponding plain text description given in the Context, try to answer user's question based on these. For structured data, in each line, the attribute name(header) is on the left of *:*, then corresponding attribute value is on the right.

### Context:
structured data {tableData}

plain text description {pt}

### User:
Limit your answer under 100 words and don't repeat the context or any info you are given. Please read the above provided structured data and it's corresponding plain text description in context and just answer the given question. Question: {question}'\
### Response:

"""




# for q4,8
sqlquery_template = 'Given an input question, first create a syntactically correct {dialect} query to run, then look at the results \
of the query and return the answer. Strict your query to a short one and dont give a long answer. *Never* use limitation to limit your query like: LIMIT {top_k} except user asked for certain row.\nWhen no specific column names are given, you can check for the answer in all columns using "OR" operator.\n\n\
Unless exactly match is required by user, use LIKE other than = in the query\n\
Never sort the results. If user asks for certain row, use LIMIT operator!\n\
Never give a sql that will return all content in the table if not explicitly asked\n\
Only give one query ended with \';\' eachtime!\n\
Carefully check the statement after WHERE clause, don\'t mix up column_name with user\'s query string, and keep the string integral for matching!\n\
When using LIKE operator, note to put column names on the left and query string on the right, don\'t reverse it\n\
Don\'t forget to append ; at the end of query and no order is needed!\n\nPay attention to use only the column \
names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay \
attention to which column is in which table.\n\nUse the following format:\n\nQuestion: Question here\nSQLQuery: SQL \
Query to run\nSQLResult: Result of the SQLQuery\nAnswer: Final answer here\n\nOnly use the \
following tables:\n{table_info}\n\nQuestion: {input}'


"""
# for q2
sqlquery_template = 'Given an input question, first create a syntactically correct {dialect} query to run, then look at the results \
of the query and return the answer. Only generate a single short query statement to run. *Never* use limitation to limit your query like: LIMIT {top_k}.\nThe query you generate should check all columns using OR operator.\n\n\
The OR operator should involve all columns you can see, don\'t only choose several columns by yourself\n\
Unless exactly match is required by user, use LIKE other than = in the query\n\
Don\'t forget to append ; at the end of query\n\nPay attention to use only the column \
names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay \
attention to which column is in which table.\n\nUse the following format:\n\nQuestion: Question here\nSQLQuery: SQL \
Query to run\nSQLResult: Result of the SQLQuery\nAnswer: Final answer here\n\nOnly use the \
following tables:\n{table_info}\n\nQuestion: {input}'
"""

def table_extraction_n_sqlQA(current_paper_tables, model_name, query_variant_list, additional_question=None, llm=None, llm_qa=None, show_errors=True):

    df_list = []
    for c_table in current_paper_tables:
        try:
            df_list.append(pd.read_csv(c_table, header=None))
        except Exception as e:
            print(f"current table is invalid: {c_table}")
    valid_df_tables = []
    variant_insource = False
    
    for df in df_list:
        c_len = 0
        for idx,row in df.iterrows():
            for row_i in range(len(row)):
                c_len += len(str(row[row_i]))


        c_num_col = len(df.axes[1])
        c_len /= len(list(df.iterrows()))

        #print(f"num_col: {c_num_col}; avg col_len= {c_len/c_num_col}")

        #print("-------------------------------")
        if True:#c_len <= 80 and c_len >= 15: # filter extracted tables by avg row_len (large: plain text block, tiny: nonsense piece)
            valid_df_tables.append(df)
            for c_variant in query_variant_list:
                for col in df.columns:
                    insource_result = df[df[col].astype(str).str.contains(c_variant, regex= True, na=False)]
                    if len(insource_result) > 0:
                        variant_insource = True
                        #print(insource_result)
                        break
                if variant_insource:
                    break
            #print(df)
    table_chunks = math.ceil(len(valid_df_tables) / 5)
   
    basic_query_answers_list = []

    for c_chunk in range(table_chunks):
        shift = c_chunk*5
        conn = sqlite3.connect('PDFpaper_Table_extractions.db')
      
        delete_all_tables(conn)

        for df_idnex, df in enumerate(valid_df_tables[shift:shift+5 if c_chunk<table_chunks-1 else len(valid_df_tables)]):
            df.to_sql(f"current_paper_table_{df_idnex}", conn, if_exists='replace')
        
            """
            #Creating a cursor object using the cursor() method
            cursor = conn.cursor()
            #Doping EMPLOYEE table if already exists
            cursor.execute("DROP TABLE emp")

            print("Table dropped... ")
            #Commit your changes in the database
            conn.commit()
            """
        current_end_index = df_idnex
        basic_query_answers = {}

        query_variant_list = query_variant_list#"c.8559-2A>G" #"p.Asn346His" # c.104del # c.516G>C

        sqlite_db_path = "./PDFpaper_Table_extractions.db"
        db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

        prompt = sqlquery_template.replace("{dialect}", db.dialect)
        prompt = PromptTemplate.from_template(prompt)

        #  return_direct: return the sql result directly
        #@func_set_timeout(180)#设定函数超执行时间_
        db_chain = SQLDatabaseChain.from_llm(llm[0], db, prompt=prompt, use_query_checker=False, verbose=False, return_intermediate_steps=True,  return_direct=True)

        c_unique_rows = []
        for c_vindx, query_variant in enumerate(query_variant_list):
            for idx in range(current_end_index+1):
                current_table = "current_paper_table_" + str(idx)
                current_table_source_index = idx + shift

                ## basic QA
                q0 = f"get the first row in table {current_table}? (take the result given by SQLResult:)"
                try:
                    
                    result0 = wrapper(db_chain,q0)
                    
                  
                except Exception as e:
                    if show_errors:
                        print(f"[ERROR] error when running question-0 on table {current_table}: {e}")
                    result0 = None
                except func_timeout.exceptions.FunctionTimedOut:
                    result0 = None
                    del db_chain; llm.clear();
                    llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                    db_chain = SQLDatabaseChain.from_llm(llm[0], db, prompt=prompt, use_query_checker=False, verbose=False, return_intermediate_steps=True,  return_direct=True)


                ## basic QA
                q1 = f"how many rows in table {current_table}? (take the result given by SQLResult:)"
                try:
                    result1 = wrapper(db_chain,q1)
                   
                except Exception as e:
                    if show_errors:
                        print(f"[ERROR] error when running question-1 on table {current_table}: {e}")
                    result1 = None
                except func_timeout.exceptions.FunctionTimedOut:
                    result1 = None
                    del db_chain; llm.clear();
                    llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                    db_chain = SQLDatabaseChain.from_llm(llm[0], db, prompt=prompt, use_query_checker=False, verbose=False, return_intermediate_steps=True,  return_direct=True)

                q2 = f"search for the string: '{query_variant}' through every column in table {current_table} using OR? (find all, no limit, column names should be like 0,1,2 as u can see in the schema)"
                try:
                    result2 = wrapper(db_chain,q2)
                   
                except Exception as e:
                    if show_errors:
                        print(f"[ERROR] error when running question-2 on table {current_table}: {e}")
                    result2 = None
                except func_timeout.exceptions.FunctionTimedOut:
                    result2 = None
                    del db_chain; llm.clear();
                    llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                    db_chain = SQLDatabaseChain.from_llm(llm[0], db, prompt=prompt, use_query_checker=False, verbose=False, return_intermediate_steps=True,  return_direct=True)
                
                q3 = f"Question: find all rows that contain the string '{query_variant}' in any column (don\'t only consider one column) (check all columns in table {current_table}) (find all, no limit)"
                try:
                    result3 = wrapper(db_chain,q3)
                 
                except Exception as e:
                    if show_errors:
                        print(f"[ERROR] error when running question-3 on table {current_table}: {e}")
                    result3 = None
                except func_timeout.exceptions.FunctionTimedOut:
                    result3 = None
                    del db_chain; llm.clear();
                    llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                    db_chain = SQLDatabaseChain.from_llm(llm[0], db, prompt=prompt, use_query_checker=False, verbose=False, return_intermediate_steps=True,  return_direct=True)

                q4 = f"Question: find all the rows that contain {query_variant} (query all columns in table {current_table} using OR) (find all, no limit)"
                try:
                    result4 = wrapper(db_chain,q4)
                   
                except Exception as e:
                    if show_errors:
                        print(f"[ERROR] error when running question-4 on table {current_table}")
                    result4 = None
                except func_timeout.exceptions.FunctionTimedOut:
                    result4 = None
                    del db_chain; llm.clear();
                    llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                    db_chain = SQLDatabaseChain.from_llm(llm[0], db, prompt=prompt, use_query_checker=False, verbose=False, return_intermediate_steps=True,  return_direct=True)

                basic_query_answers[current_table] = [(q1,result1["intermediate_steps"][-1] if result1 is not None else "sql error"),(q2,result2["intermediate_steps"][-1] if result2 is not None else "sql error"),(q3,result3["intermediate_steps"][-1] if result3 is not None else "sql error"), (q4,result4["intermediate_steps"][-1] if result4 is not None else "sql error")]
                all_results = [result2, result3, result4]
              
                c_chunk_useful_answers = []

                max_row_count = 0
                
                for c_result in all_results:
                    if c_result is not None and len(c_result["result"]) > 0:
                        c_json_string = c_result["result"]
                       
                        c_list_results = ast.literal_eval(c_json_string.strip())
                        
                        len_cl = len(c_list_results)
                        
                        for i in range(len_cl-1, -1, -1):
                            c_in = False
                            c_rowstr = json.dumps(c_list_results[i])
                            for c_content in c_list_results[i]:
                                if query_variant in str(c_content):
                                    if abs(len(query_variant) - len(str(c_content).strip())) >= 2:
                                        if not is_number(str(c_content).strip()):
                                            c_have_digits_0 = re.findall(rf"\d+{query_variant}\d+",str(c_content))
                                            c_have_digits_1 = re.findall(rf"{query_variant}\d+",str(c_content))
                                            c_have_digits_2 = re.findall(rf"\d+{query_variant}",str(c_content))
                                            c_single = re.findall(rf" {query_variant} ",str(c_content))
                                            if len(c_single) == len(c_have_digits_0) == len(c_have_digits_1) == len(c_have_digits_2) == 0:
                                                if c_rowstr not in c_unique_rows and c_vindx == 0: # remove duplicate result from different querying variant_format
                                                    c_unique_rows.append(c_rowstr)
                                                elif c_rowstr in c_unique_rows and c_vindx == 1: # remove duplicate result from different querying variant_format
                                                    break
                                                c_in = True
                                                break
                            if not c_in:
                                del c_list_results[i]

                        c_rows_count = len(c_list_results) # choose max result of all similar questions
                        if not max_row_count < c_rows_count:
                            continue

                        if result0 is not None:
                            r0_list = ast.literal_eval(result0["result"].strip())[0]
                            try:
                                r0_list = list(map(lambda x: x[1] + str(r0_list[:x[0]].count(x[1]) + 1) if r0_list.count(x[1]) > 1 else x[1], enumerate(r0_list)))
                            except Exception as e:
                                r0_list = None
                            
                            for c_index in range(len(c_list_results)):
                                if r0_list is None:
                                    break
                                if len(r0_list) == len(c_list_results[c_index]):

                                    c_list_results[c_index] = dict(zip(r0_list, c_list_results[c_index]))
                                    sub_dict = dict([(key, c_list_results[c_index][key]) for key in list(c_list_results[c_index].keys())[1:]])
                                    json_tablerow = json.dumps(sub_dict, indent=2)
                                    c_question_summary = "rephrase and describe it in plain text"
                                    c_question = "only list the existed variants/mutations in context in the following format *PatientID:... Variant:...*\nif no patient is explicitly mentioned put *PatientID:None* and don't mix up with variants/mutations. If no variants/mutations is explicitly mentioned put *Variant:None*."
                                    
                                    i_pre = int(list(c_list_results[c_index].values())[0]) - 1;
                                    i_next = int(list(c_list_results[c_index].values())[0]) + 1;
                                    json_list_answers = []
                                    if i_pre > 0:
                                        pre_row = get_row(conn, current_table, i_pre)[0]
                                        if str(pre_row[1]) == str(list(c_list_results[c_index].values())[1]):
                                            json_list_answers.insert(0,dict(zip(r0_list[1:], list(pre_row)[1:])))
                                        #print(f"pre_row: {pre_row}")

                                    next_row = get_row(conn, current_table, i_next)
                                    if len(next_row) > 0:
                                        #print(f"next_row: {next_row[0]}")
                                        next_row = next_row[0]
                                        if str(next_row[1]) == str(list(c_list_results[c_index].values())[1]):
                                            json_list_answers.append(dict(zip(r0_list[1:], list(next_row)[1:])))

                                    #json_tablerow = json.dumps(json_list_answers, indent=2)
                                    try:
                                        c_text = table2text(llm[0], str(json_tablerow), c_question)
                                        c_text_sum = table2text(llm[0], str(json_tablerow), c_question_summary)
                                    except Exception as e:
                                        
                                        c_text = None
                                        c_text_sum = None
                                        llm.clear();
                                        llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                                    except func_timeout.exceptions.FunctionTimedOut:
                                        
                                        #print("func_timeout.exceptions.FunctionTimedOut")
                                        #print("str(json_tablerow)",str(json_tablerow),"c_question",c_question)
                                        c_text = None
                                        c_text_sum = None
                                        llm.clear();
                                        print("re-loading ollama")
                                        llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))

                                        # try again
                                        try:
                                            c_text = table2text(llm[0], str(json_tablerow), c_question)
                                            c_text_sum = table2text(llm[0], str(json_tablerow), c_question_summary)
                                        except func_timeout.exceptions.FunctionTimedOut:
                                            c_text='LLM running failed'
                                            c_text_sum = "LLM running failed"
                                            llm.clear();
                                            print("re-loading ollama")
                                            llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                                    if c_text is not None:
                                        c_list_results[c_index]["plainText"] = f"## TableLLM Identified Record   \n**Source**: - Table {current_table_source_index+1} - Row {c_list_results[c_index][0]}  \n- **LLM extracted Variant/Genotypes with PatientID**： " + c_text +f"  \n- **LLM Translated Row Summary**: {c_text_sum}  " + f"  \n- **Source Row Details**: {str(json_tablerow)}  "
                                    if c_text is not None:
                                        
                                        for c_tableRow in json_list_answers:
                                            c_tableRow = json.dumps(c_tableRow, indent=2)
                                           
                                            try:
                                                c_text = table2text(llm[0], str(c_tableRow), c_question)
                                                c_text_sum = table2text(llm[0], str(c_tableRow), c_question_summary)
                                            except Exception as e:
                                                
                                                c_text = None
                                                c_text_sum = None
                                                llm.clear();
                                                llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                                            except func_timeout.exceptions.FunctionTimedOut:
                                                c_text = None
                                                c_text_sum = None
                                                llm.clear();
                                                llm.append(Ollama(model=model_name, temperature=0.0, top_p = 0.9))
                                            if c_text is not None:
                                                c_list_results[c_index]["plainText"] += f"\n### Adjacent rows potentially contain intrans variant: \n- **LLM extracted Variant/Genotypes with PatientID**： " + c_text +f"  \n- **LLM Translated Row Summary**: {c_text_sum}  " + f"  \n- **Source Row Details**: {str(c_tableRow)}  "



                                    
                                    else: c_list_results[c_index]["plainText"] = "GG"
                        

                        #print(max_row_count, c_rows_count)
                        if max_row_count < c_rows_count:
                            max_row_count = c_rows_count
                            #c_chunk_useful_answers.append((c_result["intermediate_steps"][-2],c_json_string.strip()))
                            c_chunk_useful_answers = (c_result["intermediate_steps"][-2],c_list_results)



                ## given QA
                if type(additional_question) is list:
                    for c_question in additional_question:
                        c_question = c_question + f" (only in table {current_table})"
                        try:
                            result = db_chain(c_question)
                            basic_query_answers[current_table].append((c_question,result["intermediate_steps"][-1]))
                        except Exception as e:
                            if show_errors:
                                print(f"error when running given question on table {current_table}: {e}")
                            basic_query_answers[current_table].append((c_question,["invalid_question!"]))

                elif type(additional_question) is str:
                    c_question = additional_question + f" (check all columns in table {current_table})"
                    try:
                        result = db_chain(c_question)
                        basic_query_answers[current_table].append((c_question,result["intermediate_steps"][-1]))
                    except Exception as e:
                        if show_errors:
                            print(f"error when running given question on table {current_table}: {e}")
                        basic_query_answers[current_table].append((c_question,["invalid_question!"]))

               
                if len(c_chunk_useful_answers) > 0:
                    basic_query_answers_list.append(c_chunk_useful_answers)

    return [basic_query_answers_list, variant_insource]

