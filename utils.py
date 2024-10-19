import pandas as pd
from bioc import biocxml
from bs4 import BeautifulSoup
import re
from itertools import product

import requests




def translateProtein2SingleChar(proteinNotation):
    print("start translateProtein2SingleChar")
    print("proteinNotation",proteinNotation)
    try:
        r = requests.get(f'https://mutalyzer.nl/api/normalize/{proteinNotation}?only_variants=false')
   
        j = r.json()
        return j['equivalent_descriptions']['p'][0]['description'].split(":")[-1]
    except Exception as e:
        return None
    
    

def sortVariantAlias(variantAlias,documents):
    
    for page in source_doc:
        c_rsids = re.findall(current_variant,page.page_content)
    

def table_to_2d(table_tag):
    rowspans = []  # track pending rowspans
    rows = table_tag.find_all(['tr'])
    # first scan, see how many columns we need
    colcount = 0
    for r, row in enumerate(rows):
        cells = row.find_all(['td', 'th'], recursive=False)
       
        '''
        colcount = max(
            colcount,
            sum(int(c.get('colspan', 1)) or 1 for c in cells[:-1]) + len(cells[-1:]) + len(rowspans))
        '''
        
        colcount = max(colcount,sum(int(c.get('colspan', 1))  for c in cells)  + len(rowspans))
        # update rowspan bookkeeping; 0 is a span to the bottom. 
        rowspans += [int(c.get('rowspan', 1)) or 1 or len(rows) - r for c in cells]
        rowspans = [s - 1 for s in rowspans if s > 1]

    # it doesn't matter if there are still rowspan numbers 'active'; no extra
    # rows to show in the table means the larger than 1 rowspan numbers in the
    # last table row are ignored.

    # build an empty matrix for all possible cells
    table = [[None] * colcount for row in rows]

    # fill matrix from row data
    rowspans = {}  # track pending rowspans, column number mapping to count
    for row, row_elem in enumerate(rows):
        span_offset = 0  # how many columns are skipped due to row and colspans 
        for col, cell in enumerate(row_elem.find_all(['td', 'th'], recursive=False)):
            # adjust for preceding row and colspans
            col += span_offset
            while rowspans.get(col, 0):
                span_offset += 1
                col += 1

            # fill table data
            rowspan = rowspans[col] = int(cell.get('rowspan', 1)) or len(rows) - row
            colspan = int(cell.get('colspan', 1)) or colcount - col
            # next column is offset by the colspan
            span_offset += colspan - 1
            value = cell.get_text()
            for drow, dcol in product(range(rowspan), range(colspan)):
                try:
                    table[row + drow][col + dcol] = value
                    rowspans[col + dcol] = rowspan
                except IndexError:
                    # rowspan or colspan outside the confines of the table
                    pass

        # update rowspan bookkeeping
        rowspans = {c: s - 1 for c, s in rowspans.items() if s > 1}

    return table


def convert2DF(xml_data):
    '''
    input: xml_data: the xml component of a table

    return: the converetd dataframe, collapse the table into 2D structure
    
    '''
    # parse XML string with BeautifulSoup
    xml_data = re.sub(r'\\x..', '', xml_data)

    #print(xml_data)
    soup = BeautifulSoup(xml_data, 'lxml')

    # find the table in the soup
    table = soup.find('table')
   
    # if thead and tbody 
    heads = table.find('thead')
    if(not heads):
        # if no tags of thead in the table, lets make the head as the first row
        new_body = table_to_2d(table)
        df = pd.DataFrame(new_body[1:], columns=new_body[0])
        
        return df
    new_heads = table_to_2d(heads)

    # merge multiple row of heads by concatenate unique values
    new_heads = [['' if value is None else value for value in row] for row in new_heads]
    unique_columns = [list(set(column)) for column in zip(*new_heads)]
    my_merged_header = [' '.join(column) for column in unique_columns]



    bodies = table.find('tbody')
 
    data = []
    new_body_data = table_to_2d(bodies)
    for index,cur_body_row in enumerate(new_body_data):
      
        tmp_row = ['']*len(cur_body_row)
        cur_body_row = ['' if value is None else value for value in cur_body_row]

        for k, cur_col in enumerate(cur_body_row):
            tmp_row[k] = cur_col.replace("\\n",'')
        
        data.append(tmp_row)
    df = pd.DataFrame(data, columns=my_merged_header)
    return df
def extractTablesFromXML(XML_path):

    '''
    
    XML_path: the path of the XML paper file

    return a list of dataframes, each df represents one table
    
    '''
    all_tables = []
    with biocxml.iterparse(XML_path) as reader:
        for document in reader:
            for i in range(len(document.passages)):
                if(document.passages[i].infons['type']!='table'):
                    continue
                cur_table_xml = document.passages[i].infons['xml']
                table_name = document.passages[i].infons['id']
                
                df = convert2DF(cur_table_xml)
                
                all_tables.append(df)
                
    
            
    return all_tables
    pass

def reduceIntransDuplicates(text_in_trans_list):

    formated_in_trans_list = []
    for cur_answers in text_in_trans_list:
        if('none' in cur_answers.lower() and 'contain' not in cur_answers.lower()):
            continue
        else:
            formated_in_trans_list.extend(cur_answers.split(','))
    formated_in_trans_list = list(set(formated_in_trans_list))
    return formated_in_trans_list
    pass
if __name__ == "__main__":
    print("test extractTablesFromXML")
    test_path = "/autofs/bal36md0/smli/smli/LLM-genome-curation/literatures/subset_report_XML/36546626.xml"
    table_list = extractTablesFromXML(test_path)
    print(len(table_list))