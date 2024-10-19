from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.globals import set_verbose, set_debug
import requests

from bioc import biocxml
# Import the following stuff for implementing custom retrievers
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from table_functions import table_extraction_n_sqlQA
from utils import extractTablesFromXML

set_debug(False)

# the second item is the number of returned chunks
# the third item is the abbreviated protein change notation
retriever_OK = [ False, 0, None ]


import textwrap
import os
import time
from argparse import ArgumentParser
import sys
import glob
import json
import re
import tempfile


os.environ['CURL_CA_BUNDLE'] = ''  # Fix SSL error for Mutalyzer3

PROTEIN_MAPPING_FILE = './protein.txt'

# enum types
VARIANT_QUERY = 0
INTRANS_QUERY = 1
C_VARIANT = 0  # c.123A>G
P_VARIANT = 1  # protein change


from func_timeout import func_set_timeout
import func_timeout


class VariantSpecificRetriever(BaseRetriever):
    documents: List[Document]
    k: int
    protein_map: Dict[str, str]
    
    # Assumes "query" to be the target variant (in HGVS notation)
    def _get_relevant_documents(self, query):
        variant = query
        # Remove the contig name (NM_xxxxxx)
        target_var = variant.split(":")[-1]
        # Remove c.() from the variant notation (by default is c.(123A>G) or c.123A>G)
        var_dna = target_var.replace('c.', '').replace('(', '').replace(')', '')
        # Translate the mutation to protein change using Mutalyzer
        var_protein = None
        var_protein_short = None
        try:
            r = requests.get(f'https://mutalyzer.nl/api/normalize/{variant}?only_variants=false')
            j = r.json()
            returned_prot = j['protein']['description'].split(':')[-1]
            # Remove the p.()
            m = re.match(r'p.\((.*)\)', returned_prot)
            prot = m.group(1)
            if len(prot) < 5:  # Too short (sometimes Mutalyzer returns something like p.(=) )
                raise Exception(f'Protein change too short: {returned_prot}')
            # Sometimes the protein mutation is like Cys1447Glnfs29 but some papers write as Cys1447fs,
            # so we remove the whole Glnfs part
            var_protein = re.sub(r'[A-Za-z]{3}fs.*', '', prot)
            # Convert the protein to short form ( -> )
            var_protein_short = var_protein
            for (k,v) in self.protein_map.items():
                var_protein_short = var_protein_short.replace(k, v)
            # Remove X and * (meaning Terminal) from the protein notation, since we don't know the paper is using which one
            var_protein = var_protein.replace('X', '').replace('*', '')
            var_protein_short = var_protein_short.replace('X', '').replace('*', '')
            #print(f'Protein : {var_protein} ({var_protein_short})')
        except KeyError as e:
            #print('Protein: [ERROR] Not found by Mutalyzer')
            pass
        except Exception as e:
            #print(f'Protein : [ERROR] {e}')
            pass
        
        # Done with conversion. Now do the retrieval (= regex matching)

        retrieved_chunks = []
        # Construct the regex pattern for DNA:
        # 1. 123A>G becomes \s*123\s*A>G (allow spaces around numbers)
        # 2. Further becomes \s*123\s*123A\s*>\s*G (allow spaces around > )
        dna_pattern = re.sub('([0-9]+)', '\\\\s*\\1\\\\s*', re.escape(var_dna))
        dna_pattern = re.sub('(>)', '\\\\s*\\1\\\\s*', dna_pattern)
        dna_matcher = re.compile(dna_pattern, re.IGNORECASE)
        for chunk in self.documents:
            # Re-encode the text to get rid of those annoying Unicode \x80\x89 (whitespaces)
            text = chunk.page_content.encode('utf-8').decode('unicode_escape').encode('latin-1').decode('utf-8')
            if dna_matcher.search(text) or \
               var_protein and chunk.page_content.find(var_protein) >= 0 or \
               var_protein_short and chunk.page_content.find(var_protein_short) >= 0:
                retrieved_chunks += [ chunk ]
        # If neither DNA nor protein change could retrieve anything,
        # we resort to matching by positions only...
        if not retrieved_chunks:
            dig_dna = re.findall(r'\d+', var_dna)
            dig_protein = re.findall(r'\d+', var_protein) if var_protein else None
            dig_dna_matcher = re.compile('\D' + str(dig_dna[0]) + '\D') if dig_dna else None
            dig_protein_matcher = re.compile('\D' + str(dig_protein[0]) + '\D') if dig_protein else None
            for chunk in self.documents:
                if dig_dna_matcher and dig_dna_matcher.search(chunk.page_content) or \
                   dig_protein_matcher and dig_protein_matcher.search(chunk.page_content):
                   retrieved_chunks += [ chunk ]
        if len(retrieved_chunks) > 0:
            retriever_OK[0] = True
        retriever_OK[1] = len(retrieved_chunks)
        retriever_OK[2] = var_protein_short
        return retrieved_chunks[:self.k]


# Load the protein abbreviatioon map from a file
def load_protein_map(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    m = { x.split()[0]: x.split()[1] for x in lines }
    return m


# Load paper from XML file
def load_xml_paper(filename, filter_tables=False):
    out_doc = ''
    with open(filename, 'r', encoding='utf8') as fp:  # better use utf8
        collection = biocxml.load(fp)
    document = collection.documents[0]
    for passage in document.passages:
        section_type = passage.infons.get('section_type', '').upper()
        if filter_tables and section_type in [ 'TABLE', 'REF', 'COMP_INT', 'AUTH_CONT', 'SUPPL' ]:
            pass  # filter away this section
        else:
            out_doc += passage.text + '\n'
    return out_doc



template_PM3_answer_chain_llama3 = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a specialist in biogenetics, answer only based on user's input!<|eot_id|>
<|start_header_id|>user<|end_header_id|>
The variant in HGVS format is {question}, don't include this in your answer if condisering compound het variants.
Given the context: '{context}' and target variant {c_variant}. Answer the question: {proposedQuestion}<|eot_id|>.
<|start_header_id|>assistant<|end_header_id|>\n
"""



def split_docs(documents,chunk_size=1500,chunk_overlap=100):
# Responsible for splitting the documents into several chunks
    
    # Initializing the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)
    
    # returning the document chunks
    return chunks





# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt, "verbose": False} # customizing the prompt
    )




# Prettifying the response
@func_set_timeout(300)
def get_answers_PM3(query, chain):

    # Getting response from chain
    input_dict = {'query': query}

    response = chain(input_dict)
    
    return response

def loadTextModel(model_name):
    print("Loading model",model_name)
    if "llama3" in model_name:
        llm_a = Ollama(model=model_name,temperature=0.0, top_p = 0.9, stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"])
    elif model_name == "phi3":

        llm_a = Ollama(model=model_name,temperature=0.0, top_p = 0.9, stop=["<|user|>","<|assistant|>","<|system|>","<|end|>","<|endoftext|>", "<|reserved_special_token"])
    else:
        llm_a = Ollama(model=model_name,temperature=0.0, top_p = 0.9)
    print("Loading model DONE")
    return llm_a

def main():
    parser = ArgumentParser(description='AutoPM3')
    parser.add_argument(
        '--model_name_text',
        help="llm used for answering generated questions",
        required=False,
        default='llama3_loraFT-8b-f16',
    )
    parser.add_argument(
        '--model_name_table',
        help="llm used for table queries",
        required=False,
        default='sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp.Q8_0',
    )
    parser.add_argument(
        '--query_variant',
        help="query variant in HGVS format",
        required=True,
    )
    parser.add_argument(
        '--paper_path',
        help="paper_path of the query literature",
        required=True,
    )


    # print help message if no argument input
    if len(sys.argv) <= 1 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        parser.print_help(sys.stderr)
        sys.exit(0)

    args = parser.parse_args()   
    results = query_variant_in_paper_xml(args.query_variant,args.paper_path,args.model_name_table,args.model_name_text)
    print(results)


def query_variant_in_paper_xml(query_variant, xml_path, model_name_table, model_name_text):
    
    llm_a = loadTextModel(model_name_text)
    llm_table = [Ollama(model=model_name_table, temperature=0.0, top_p=0.9) ]

    # Read protein abbreviation table
    protein_map = load_protein_map(PROTEIN_MAPPING_FILE)

    # check if the query variant is in the correct format (TODO)
    c_variant = query_variant.split(":")[-1]

    # Check if the paper (XML, PDF) exists
    xml_fn =xml_path

    if not os.path.exists(xml_fn):
        print('XML paper not found. Abort.')
        sys.exit(-1)

    # Load the XML paper and filter away tables and useless sections,
    # then split into chunks 
    doc_filtered = load_xml_paper(xml_fn, filter_tables=True)
    doc_wrapper = [ Document(page_content = doc_filtered, metadata = {'source': 'local'}) ]
    doc_chunks = split_docs(doc_wrapper)
    # Try our custom retriever
    variant_retriever = VariantSpecificRetriever(documents=doc_chunks, k=5, protein_map=protein_map)
    variant_hgvs = query_variant

    try:
        r = requests.get(f'https://mutalyzer.nl/api/normalize/{query_variant}?only_variants=false')
        j = r.json()
        protein = j['protein']['description'].split(':')[-1]
        if protein == 'p.(=)':
            raise Exception('invalid notation')
      
        c_protein_id = re.findall(r"\d+",protein)
    except Exception as e:
       
        protein = None
        protein_short = None


    ##########################
    # Do table queries first #
    ##########################
    table_src_contains_variant = False
    table_query_results = []

    # Find all the table CSV files for this PMID
    #relevant_tables = [ f for f in table_csv_files if str(c_pmid) in f ]
    relevant_tables = extractTablesFromXML(xml_fn)
    c_variant_id = None
    c_max = 0
    c_tmp_digit = re.findall(r"\d+",c_variant)
    for c_digit in c_tmp_digit:
        if len(c_digit) > c_max: # 3
            c_variant_id = c_digit
            c_max = len(c_digit)
   
    if relevant_tables:
        variant_alias = [c_variant_id, c_protein_id[0]] if protein is not None and len(c_protein_id) > 0 else [c_variant_id]
    
        csv_files = []
        csv_filenames = []
        # Write the extracted tables to temporary CSV files
        for table in relevant_tables:
            
            tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=True)
            csv_files.append(tmpfile)
            csv_filenames.append(tmpfile.name)
            #print("temp tables",tmpfile.name)
            table.to_csv(tmpfile.name,index=False)
        table_query_return = table_extraction_n_sqlQA(csv_filenames, model_name_table,
            query_variant_list=variant_alias, llm=llm_table, llm_qa=llm_table, show_errors=False)
     
        # Close and delete the temp files
        for tmpfile in csv_files:
            tmpfile.close()

        if table_query_return is None:
            
            pass
            #print('[ERROR] Something went wrong with the tables.')
        else:
            ( table_query_results, table_src_contains_variant ) = table_query_return[:2]
            # Collect the answer candidates from the returned results
            table_results_plaintext = []
            for c_cmd in table_query_results:
                for c_answer in c_cmd[1]:
                    if not isinstance(c_answer, tuple):
                        try:
                            table_results_plaintext.append(c_answer['plainText'])
                        except Exception as e:
                            pass
    else:
        table_results_plaintext = ["No table found"]
    ###########################
    # Now do the text queries #
    ###########################

    # We use "protein" instead of "c_protein_id[0]" for text.
    #   - "protein": The protein change returned by Mutalzyer
    #   - "c_protein_id[0]": Only the digits in "protein"
    
    variant_alias = [c_variant, protein] if protein is not None and len(c_protein_id) > 0 else [c_variant]

    text_variant_hit = False
    text_intrans_list = []
    text_src_contains_variant = False
    text_variant_answer = ""

    MAX_RETRIES = 1  # number of retries before giving up

    PM3_answer_prompt = PromptTemplate.from_template(template_PM3_answer_chain_llama3)
    for c_index, current_variant in enumerate(variant_alias):
        
        for query_type in range(2):  # variant query, in-trans query
            retriever_OK[0] = False
            if query_type == VARIANT_QUERY:
                my_predefined_query = f"Does the paper mention the queried variant ({current_variant}) and what is the surrounding context?" + f"""if such variant is existed, say *YES* at first otherwise say *None* (focus on variant: {current_variant})"""
            elif query_type == INTRANS_QUERY:
                my_predefined_query = f"If {current_variant} is compound heterozygous with another variant, name it; if {current_variant} is homozygous, say homozygous; if no related variant is found, say *None*. List all results seperated by comma and wrap the answers by *."""

            num_retries = 0
            query_success = False
            while not query_success and num_retries <= MAX_RETRIES:
                try:
                    PM3_answer_chain = load_qa_chain(variant_retriever, llm_a, PM3_answer_prompt.partial(proposedQuestion=my_predefined_query, c_variant=current_variant))
                    cur_answers_all = get_answers_PM3(variant_hgvs, PM3_answer_chain)
                    if not retriever_OK[0]:
                        break
                    else:
                        text_src_contains_variant = True
                    protein_short = retriever_OK[2]
                    cur_answers = cur_answers_all['result']
                    query_success = True
                except func_timeout.exceptions.FunctionTimedOut:
                    del llm_a;
                    llm_a = loadTextModel(model_name_text)
                    num_retries += 1

            if not query_success:
                continue

            # Wrapping the text for better output in Jupyter Notebook
            wrapped_text = textwrap.fill(cur_answers_all['result'], width=100)
            
            if query_type == VARIANT_QUERY:
                source_doc = cur_answers_all['source_documents']
                c_variant_inRetrieved = False
                for page in source_doc:
                    c_rsids = re.findall(current_variant if c_index == C_VARIANT else c_protein_id[0], page.page_content)
                    if len(c_rsids) > 0:
                        c_variant_inRetrieved = True
                if 'yes' in cur_answers.lower():
                    text_variant_hit = True
                    if c_index == C_VARIANT:
                        text_variant_answer = "\n- **[DNA match result]**:"+cur_answers
                    else:
                        text_variant_answer += "\n- **[Protein match result]**:"+cur_answers
                elif text_variant_answer == "":
                    text_variant_answer = "\n- **Variant not found in text part!**"

            elif query_type == INTRANS_QUERY:
                if "none" not in cur_answers.lower() or "contain" in cur_answers.lower():
                    text_intrans_list.append(cur_answers)
                #text_intrans_list.append(cur_answers)

      
    table_results_plaintext_output = [str(xx).strip("\n") + "\n\n" for xx in table_results_plaintext]
    #print(f'# Output Summary:  \n  \n## **Query Variant and Relative Intrans-variant/Genotype Found in PaperTables**:  \n{"".join([str(xx) for xx in table_results_plaintext_output])}  \n  \n## **Query Variant Found in PaperText**:  \n- {text_variant_answer if text_variant_hit != "" else "Variant not found in text part!"} \n  \n## **Query Variant\'s Intrans-variant Found in PaperText**:  \n{text_intrans_list if text_variant_answer != "Variant not found in text part!" and text_intrans_list else "None!"}')
    results = f'# Output Summary:  \n  \n## **Query Variant and Relative Intrans-variant/Genotype Found in PaperTables**:  \n{"".join([str(xx) for xx in table_results_plaintext_output])}  \n  \n## **Query Variant Found in PaperText**:  \n- {text_variant_answer if text_variant_hit != "" else "Variant not found in text part!"} \n  \n## **Query Variant\'s Intrans-variant Found in PaperText**:  \n{text_intrans_list if text_variant_answer != "Variant not found in text part!" and text_intrans_list else "None!"}'
    return results


if __name__ == "__main__":
  
    main()

