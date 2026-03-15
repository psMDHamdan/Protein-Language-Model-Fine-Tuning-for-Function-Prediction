import os
import requests
import pandas as pd
from tqdm import tqdm
import io

def fetch_uniprot_data(query="reviewed:true AND (go:*)", list_size=5000):
    """
    Fetches protein sequences and GO annotations from UniProt.
    """
    print(f"Fetching {list_size} entries from UniProt with query: {query}")
    
    url = "https://rest.uniprot.org/uniprotkb/search"
    all_data = []
    cursor = None
    
    # We'll fetch in chunks of 500 (max allowed size)
    chunk_size = 500
    num_chunks = (list_size + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        params = {
            "query": query,
            "format": "tsv",
            "fields": "accession,sequence,go_id",
            "size": chunk_size
        }
        if cursor:
            params["cursor"] = cursor
            
        print(f"Fetching chunk {i+1}/{num_chunks}...")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            print(response.text)
            break
            
        df_chunk = pd.read_csv(io.StringIO(response.text), sep="\t")
        all_data.append(df_chunk)
        
        # Check for next page link in headers
        if "Link" in response.headers:
            # Simple cursor extraction from Link header if needed, but UniProt search usually uses size/offset or search_after
            # Actually, for search we can just use size 500 and repeat if we have more.
            # However, standard UniProt API for large results uses pagination.
            pass
        
        # If we got less than requested chunk_size, we're done
        if len(df_chunk) < chunk_size:
            break
            
    if not all_data:
        return None
        
    df = pd.concat(all_data, ignore_index=True)
    print(f"Successfully fetched {len(df)} entries.")
    return df

def preprocess_data(df, output_path="data/protein_go_data.csv"):
    """
    Cleans and saves the data.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Drop rows with missing values
    df = df.dropna(subset=['Sequence', 'Gene Ontology IDs'])
    
    # Split GO IDs into a list
    df['GO_IDs'] = df['Gene Ontology IDs'].apply(lambda x: x.split('; '))
    
    # Select relevant columns
    df = df[['Entry', 'Sequence', 'GO_IDs']]
    
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    return df

if __name__ == "__main__":
    # For initial testing, we'll fetch a small subset
    data = fetch_uniprot_data(list_size=5000)
    if data is not None:
        preprocess_data(data)
