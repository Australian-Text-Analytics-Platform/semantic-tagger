#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:29:44 2022

@author: sjuf9909
"""
# import required packages
import codecs
import hashlib
import io
import os
from tqdm import tqdm

# pandas: tools for data processing
import pandas as pd

# spaCy and NLTK: natural language processing tools for working with language/text data
import spacy
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# ipywidgets: tools for interactive browser controls in Jupyter notebooks
import ipywidgets as widgets
from IPython.display import clear_output, FileLink

class DownloadFileLink(FileLink):
    '''
    Create link to download files in Jupyter Notebook
    '''
    html_link_str = "<a href='{link}' download={file_name}>{link_text}</a>"

    def __init__(self, path, file_name=None, link_text=None, *args, **kwargs):
        super(DownloadFileLink, self).__init__(path, *args, **kwargs)

        self.file_name = file_name or os.path.split(path)[1]
        self.link_text = link_text or self.file_name

    def _format_path(self):
        from html import escape

        fp = "".join([self.url_prefix, escape(self.path)])
        return "".join(
            [
                self.result_html_prefix,
                self.html_link_str.format(
                    link=fp, file_name=self.file_name, link_text=self.link_text
                ),
                self.result_html_suffix,
            ]
        )
        

class SemanticTagger():
    '''
    Rule based token and Multi Word Expression semantic tagger for the English language
    '''
    
    def __init__(self):
        '''
        Initiate the SemanticTagger
        '''
        # download spaCy's en_core_web_sm, the pre-trained English language tool from spaCy
        print('Loading spaCy language model...')
        print('This may take a while...')
        
        # exclude 'parser' and 'ner' components as we do not need them
        self.nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
        
        # load the English PyMUSAS rule based tagger in a separate spaCy pipeline
        english_tagger_pipeline = spacy.load('en_dual_none_contextual')
        
        # adds the English PyMUSAS rule based tagger to the main spaCy pipeline
        self.nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
        print('Finished loading.')
        
        # initiate variables to hold texts and tagged texts in pandas dataframes
        self.text_df = None
        self.tagged_df = None
        
        # initiate the variables for file uploading
        self.file_uploader = widgets.FileUpload(
            description='Upload your files (txt, csv or xlsx)',
            accept='.txt, .xlsx, .csv ', # accepted file extension 
            multiple=True,  # True to accept multiple files
            error='File upload unsuccessful. Please try again!',
            layout = widgets.Layout(width='320px')
            )
    
        self.upload_out = widgets.Output()
        
        # give notification when file is uploaded
        def _cb(change):
            with self.upload_out:
                # clear output and give notification that file is being uploaded
                clear_output()
                
                # reading uploaded files
                self.process_upload()
                
                # clear saved value in cache and reset counter
                self.file_uploader.value.clear()
                self.file_uploader._counter=0
                
                # give notification when uploading is finished
                print('Finished uploading files.')
                print('{} text documents are loaded for tagging.'.format(self.text_df.shape[0]))
            
        # observe when file is uploaded and display output
        self.file_uploader.observe(_cb, names='data')
        self.upload_box = widgets.VBox([self.file_uploader, self.upload_out])
        
        
    def load_txt(self, value: dict) -> list:
        '''
        Load individual txt file content and return a dictionary object, 
        wrapped in a list so it can be merged with list of pervious file contents.
        
        Args:
            value: the file containing the text data
        '''
        temp = {'text_name': value['metadata']['name'][:-4],
                'text': codecs.decode(value['content'], encoding='utf-8', errors='replace')
        }
    
        return [temp]


    def load_table(self, value: dict, file_fmt: str) -> list:
        '''
        Load csv or xlsx file
        
        Args:
            value: the file containing the text data
            file_fmt: the file format, i.e., 'csv', 'xlsx'
        '''
        # read the file based on the file format
        if file_fmt == 'csv':
            temp_df = pd.read_csv(io.BytesIO(value['content']))
        if file_fmt == 'xlsx':
            temp_df = pd.read_excel(io.BytesIO(value['content']))
            
        # check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('text' not in temp_df.columns) or ('text_name' not in temp_df.columns):
            print('File {} does not contain the required header "text" and "text_name"'.format(value['metadata']['name']))
            return []
        
        # return a list of dict objects
        temp = temp_df[['text_name', 'text']].to_dict(orient='index').values()
        
        return temp


    def hash_gen(self, temp_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create column text_id by md5 hash of the text in text_df
        
        Args:
            temp_df: the temporary pandas dataframe containing the text data
        '''
        temp_df['text_id'] = temp_df['text'].apply(lambda t: hashlib.md5(t.encode('utf-8')).hexdigest())
        
        return temp_df


    def process_upload(self, deduplication: bool = True):    
        '''
        Pre-process uploaded .txt files into pandas dataframe

        Args:
            deduplication: option to deduplicate text_df by text_id
        '''
        # create an empty list for a placeholder to store all the texts
        all_data = []
        
        # read and store the uploaded files
        files = list(self.file_uploader.value.keys())
        
        print('Reading uploaded files...')
        print('This may take a while...')
        for file in tqdm(files):
            if file.lower().endswith('txt'):
                text_dic = self.load_txt(self.file_uploader.value[file])
            else:
                text_dic = self.load_table(self.file_uploader.value[file], \
                    file_fmt=file.lower().split('.')[-1])
            all_data.extend(text_dic)
        
        # convert them into a pandas dataframe format, add unique id and pre-process text
        self.text_df = pd.DataFrame.from_dict(all_data)
        self.text_df = self.hash_gen(self.text_df)
        
        # deduplicate the text_df by text_id
        if deduplication:
            self.text_df.drop_duplicates(subset='text_id', keep='first', inplace=True)
    
    
    def add_tagger(self, text: str) -> pd.DataFrame:
        '''
        add semantic tags to the texts and convert into pandas dataframe

        Args:
            text: the text to be tagged by the semantic tagger
        '''
        # clean empty spaces in the text
        text = sent_tokenize(text)
        text = ' '.join(text)
        
        # apply spacy language model to the text
        doc = self.nlp(text)
        
        # extract the semantic tag for each token
        tagged_text = [{'text':token.text,
                        'lemma':token.lemma_,
                        'pos':token.pos_,
                        'start_end_index':(token._.pymusas_mwe_indexes[0][0],
                                           token._.pymusas_mwe_indexes[0][1]),
                        'mwe':['yes' if (token._.pymusas_mwe_indexes[0][1]-token._.pymusas_mwe_indexes[0][0])>1 else 'no'][0],
                        'usas_tags':token._.pymusas_tags} for token in doc]
        
        # convert output into pandas dataframe
        tagged_text_df = pd.DataFrame.from_dict(tagged_text)
        
        return tagged_text_df
    
    
    def save_to_excel(self, 
                      tagged_text: str, 
                      file_name: str, 
                      sheet_name: str, 
                      mode: str='w', 
                      if_sheet_exists: str=None):
        '''
        save tagged text into an excel spreadsheet

        Args:
            tagged_text: the text with its semantic tags 
            file_name: file name of the output excel file 
            sheet_name: the excel sheet name for the tagged text 
            mode: write ('w') or append ('a') mode
            if_sheet_exists: what to do if a sheet with the same name already exists
        '''
        with pd.ExcelWriter(path=file_name, 
                            mode=mode, 
                            if_sheet_exists=if_sheet_exists) as writer:
            tagged_text.to_excel(writer, sheet_name=sheet_name)
            
            
    def tag_text(self, file_name: str):
        '''
        save all tagged texts into an excel spreadsheet using text_name as the sheet name

        Args:
            file_name: file name of the output excel file 
        '''
        for text in tqdm(self.text_df.itertuples(), total=len(self.text_df)):
            tagged_text = self.add_tagger(text.text)
            sheet_name = text.text_name[:15]
            try:
                self.save_to_excel(tagged_text, 
                                   file_name, 
                                   sheet_name, 
                                   'a', 
                                   'new')
            except:
                self.save_to_excel(tagged_text, 
                                   file_name, 
                                   sheet_name)
        
        print('Semantic tags successfully added and saved into {}!'.format(file_name))