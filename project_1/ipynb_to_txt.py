#!/usr/bin/env python3
"""
This script converts a Jupyter notebook (.ipynb) file to a text file (.txt),
extracting the content of all cells including markdown, code, and their outputs.
"""

import json
import sys
import os
import argparse

def convert_notebook_to_text(notebook_path, output_path=None):
    """
    Convert a Jupyter notebook to a text file.
    
    Args:
        notebook_path (str): Path to the input .ipynb file
        output_path (str, optional): Path to the output .txt file. If None, 
                                    uses the same name as the input file with .txt extension
    
    Returns:
        str: Path to the created text file
    """
    # If output path is not specified, create one based on the input file
    if output_path is None:
        base_name = os.path.splitext(notebook_path)[0]
        output_path = f"{base_name}.txt"
    
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Open the output file
        with open(output_path, 'w', encoding='utf-8') as out_file:
            # Write a header
            out_file.write(f"# Contents of {os.path.basename(notebook_path)}\n\n")
            
            # Process each cell
            for i, cell in enumerate(notebook.get('cells', [])):
                cell_type = cell.get('cell_type', '')
                source = cell.get('source', [])
                
                # Convert source to a single string if it's a list
                if isinstance(source, list):
                    source = ''.join(source)
                
                # Write cell type and number as a header
                out_file.write(f"## Cell {i+1} ({cell_type})\n\n")
                
                # Write the source content
                out_file.write(f"{source}\n\n")
                
                # If it's a code cell, include the outputs
                if cell_type == 'code' and 'outputs' in cell:
                    out_file.write("### Output:\n\n")
                    
                    for output in cell['outputs']:
                        output_type = output.get('output_type', '')
                        
                        # Handle different output types
                        if output_type == 'stream':
                            out_file.write(f"Stream ({output.get('name', '')}):\n")
                            text = output.get('text', [])
                            if isinstance(text, list):
                                text = ''.join(text)
                            out_file.write(f"{text}\n\n")
                            
                        elif output_type == 'execute_result':
                            data = output.get('data', {})
                            
                            # Text/plain is the most common, but handle other formats too
                            for mime_type, content in data.items():
                                if 'image' in mime_type:
                                    out_file.write("[Image output not shown in text format]\n\n")
                                else:
                                    out_file.write(f"Result ({mime_type}):\n")
                                    if isinstance(content, list):
                                        content = ''.join(content)
                                    out_file.write(f"{content}\n\n")
                                
                        elif output_type == 'display_data':
                            data = output.get('data', {})
                            for mime_type, content in data.items():
                                if 'image' in mime_type:
                                    out_file.write("[Image output not shown in text format]\n\n")
                                else:
                                    out_file.write(f"Display ({mime_type}):\n")
                                    if isinstance(content, list):
                                        content = ''.join(content)
                                    out_file.write(f"{content}\n\n")
                                
                        elif output_type == 'error':
                            ename = output.get('ename', '')
                            evalue = output.get('evalue', '')
                            traceback = output.get('traceback', [])
                            
                            out_file.write(f"Error: {ename}: {evalue}\n")
                            if traceback:
                                if isinstance(traceback, list):
                                    traceback = '\n'.join(traceback)
                                out_file.write(f"Traceback:\n{traceback}\n\n")
                
                # Add a separator between cells
                out_file.write("-" * 80 + "\n\n")
        
        print(f"Successfully converted {notebook_path} to {output_path}")
        return output_path
        
    except json.JSONDecodeError:
        print(f"Error: {notebook_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")
        return None

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Convert Jupyter notebooks to text files')
    parser.add_argument('notebook_path', help='Path to the input .ipynb file')
    parser.add_argument('-o', '--output', help='Path to the output .txt file (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert the notebook
    convert_notebook_to_text(args.notebook_path, args.output)

if __name__ == "__main__":
    main()