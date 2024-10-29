import streamlit as st
import os
import requests

from AutoPM3_main import query_variant_in_paper_xml


# Function to load a XML file from a URL
def load_xml(url,pmid):

    temp_paper_file_root = "./xml_papers"
    if(not os.path.exists(temp_paper_file_root)):
        os.mkdir(temp_paper_file_root)
    fn = str(pmid)+".xml"
    xml_path = os.path.join(temp_paper_file_root,fn)
    if(os.path.exists(xml_path)):
        return xml_path
    
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200 and response.headers['Content-type'] == 'text/xml':
        xml_data = response.content
        # save it to the temp dir
        with open(xml_path, 'wb') as f:
            f.write(response.content)
        return xml_path
    else:
        raise Exception('Invalid PMID. Make sure the publication has OpenAccess.')
        return None
  

# Function to display model results
def display_results(model, data):
    # Assuming 'model' is your trained model and 'data' is the input to the model
    results = model.predict(data)
    st.write(results)

# Main
st.title('AutoPM3')

if st.button('Example', type='primary'):
    st.session_state.variant_name = 'NM_004004.5:c.71G>A'
    st.session_state.pmid = '15070423'

variant_name = st.text_input('Step 1. Enter the variant (HGVS notation)', key='variant_name')

# Get the URL of the XML from the user
paper_url = ''
pmid = st.text_input('Step 2. Enter the PMID of the paper', key='pmid')
if pmid:
    try:
        pmid = int(pmid)
        paper_url = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmid}/unicode'
    except ValueError:
        st.write('Invalid PMID.')

if st.button('Run', type='primary'):
    summarized_results = ""
    if paper_url and variant_name:
        try:
            # Load and display the XML
            xml_path = load_xml(paper_url,pmid)
            summarized_results = query_variant_in_paper_xml(variant_name, xml_path, 'sqlcoder-7b-Mistral-7B-Instruct-v0.2-slerp.Q8_0', 'llama3_loraFT-8b-f16')
            # Display the summarized results
            st.write(summarized_results)
        except Exception as e:
            st.write('An error has occurred.')
            st.write(str(e))
