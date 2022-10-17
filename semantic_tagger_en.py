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
from zipfile import ZipFile
from pyexcelerate import Workbook
from collections import Counter
from pathlib import Path
import re
import joblib
import warnings
import itertools

# pandas: tools for data processing
import pandas as pd

# matplotlib: visualization tool
from matplotlib import pyplot as plt

# spaCy and NLTK: natural language processing tools for working with language/text data
import spacy
from spacy.tokens import Doc
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# langdetect: tool to detect language in a text
from langdetect import detect

# ipywidgets: tools for interactive browser controls in Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, clear_output, FileLink, HTML

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
        
        # Construction via spaCy pipeline
        exclude = ['ner', 'parser', 'entity_linker', 'entity_ruler', 
                   'morphologizer', 'transformer']
        
        # load spacy and exclude unecessary components 
        self.nlp = spacy.load('en_core_web_sm', exclude=exclude)
        print('Finished loading.')
        
        # initiate other necessary variables
        self.mwe = None
        self.mwe_count = 0
        self.text_df = None
        self.tagged_df = None
        self.large_file_size = 1000000
        self.max_to_process = 50
        self.cpu_count = joblib.cpu_count()
        self.selected_text = {'left': None, 'right': None}
        self.text_name = {'left': None, 'right': None}
        self.df = {'left': None, 'right': None}
        
        # create an output folder if not already exist
        os.makedirs('output', exist_ok=True)
        
        # get usas_tags definition
        usas_def_file = './documents/semtags_subcategories.txt'
        self.usas_tags = dict()
        
        # reading line by line
        with open(usas_def_file) as file_x:
            for line in file_x:
                self.usas_tags[line.rstrip().split('\t')[0]] = line.rstrip().split('\t')[1]
        
        # initiate the variables for file uploading
        self.file_uploader = widgets.FileUpload(
            description='Upload your files (txt, csv, xlsx or zip)',
            accept='.txt, .xlsx, .csv, .zip', # accepted file extension
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
                
                # check file size
                self.check_file_size(self.file_uploader)
                
                # reading uploaded files
                self.process_upload()
                
                # clear saved value in cache and reset counter
                self.file_uploader._counter=0
                self.file_uploader.value.clear()
                
                # give notification when uploading is finished
                print('Finished uploading files.')
                print('{} text documents are loaded for tagging.'.format(self.text_df.shape[0]))
            
        # observe when file is uploaded and display output
        self.file_uploader.observe(_cb, names='data')
        self.upload_box = widgets.VBox([self.file_uploader, self.upload_out])
        
        # CSS styling 
        self.style = """
        <style scoped>
            .dataframe-div {
              max-height: 350px;
              max-width: 1050px;
              overflow: auto;
              position: relative;
            }
        
            .dataframe thead th {
              position: -webkit-sticky; /* for Safari */
              position: sticky;
              top: 0;
              background: #2ca25f;
              color: white;
            }
        
            .dataframe thead th:first-child {
              left: 0;
              z-index: 1;
            }
        
            .dataframe tbody tr th:only-of-type {
                    vertical-align: middle;
                }
        
            .dataframe tbody tr th {
              position: -webkit-sticky; /* for Safari */
              position: sticky;
              left: 0;
              background: #99d8c9;
              color: white;
              vertical-align: top;
            }
        </style>
        """
    
    
    def select_mwe(self):
        '''
        loading spaCy language model and PyMUSAS pipeline based on the selected mwe option
        '''
        if self.mwe_count==0:
            print('Loading the semantic tagger...')
            print('This may take a while...')
            # load the English PyMUSAS rule based tagger in a separate spaCy pipeline
            if self.mwe=='yes':
                english_tagger_pipeline = spacy.load('en_dual_none_contextual') # with mwe, but much slower
            else:
                english_tagger_pipeline = spacy.load('en_single_none_contextual') # without mwe
            
            # adds the English PyMUSAS rule based tagger to the main spaCy pipeline
            self.nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
            self.nlp.add_pipe('sentencizer')
            print('Finished loading.')
            self.mwe_count+=1
        else:
            print('\nSemantic tagger has been loaded and ready for use.')
            print('Please re-start the kernel if you wish to select another option (at the top, select Kernel - Restart).')
            
    
    def mwe_widget(self):
        '''
        Create a widget to select the mwe option
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value='Would you like to include multi-word expressions (mwe)?',
            placeholder='',
            description=''
            )
        
        enter_text2 = widgets.HTML(
            value = '<b><font size=2.5>Warning:</b> including mwe extraction will make the process much slower. \
                For corpus >500 texts, we recommend choosing the non-MWE version</b>',
            #value='<b>Warning:</b> including mwe extraction will make the process much slower.',
            placeholder='',
            description=''
            )
        
        # select mwe
        mwe_selection = widgets.RadioButtons(
            options=['yes', 'no'],
            value='yes', 
            layout={'width': 'max-content'}, # If the items' names are long
            description='',
            disabled=False
            )
        
        # widget to show loading button
        mwe_button, mwe_out = self.click_button_widget(desc='Load Semantic Tagger', 
                                                       margin='15px 0px 0px 0px',
                                                       width='180px')
        
        # function to define what happens when the button is clicked
        def on_mwe_button_clicked(_):
            with mwe_out:
                clear_output()
                
                # load selected language semantic tagger
                self.mwe = mwe_selection.value
                self.select_mwe()
                
                if self.mwe=='no':
                    enter_text.value = 'Semantic tagger without MWE extraction has been loaded and ready for use.'
                    enter_text2.value=''
                    mwe_selection.options=['no']
                else:
                    enter_text.value = 'Semantic tagger with MWE extraction has been loaded and ready for use.'
                    mwe_selection.options=['yes']
                
                
        # link the button with the function
        mwe_button.on_click(on_mwe_button_clicked)
        
        vbox = widgets.VBox([enter_text, 
                             mwe_selection, 
                             enter_text2, 
                             mwe_button, mwe_out])
        
        return vbox

    
    def click_button_widget(
            self, 
            desc: str, 
            margin: str='10px 0px 0px 10px',
            width='320px'
            ):
        '''
        Create a widget to show the button to click
        
        Args:
            desc: description to display on the button widget
            margin: top, right, bottom and left margins for the button widget
        '''
        # widget to show the button to click
        button = widgets.Button(description=desc, 
                                layout=Layout(margin=margin, width=width),
                                style=dict(font_style='italic',
                                           font_weight='bold'))
        
        # the output after clicking the button
        out = widgets.Output()
        
        return button, out
    
    
    def check_file_size(self, file):
        '''
        Function to check the uploaded file size
        
        Args:
            file: the uploaded file containing the text data
        '''
        # check total uploaded file size
        total_file_size = sum([i['metadata']['size'] for i in self.file_uploader.value.values()])
        print('The total size of the upload is {:.2f} MB.'.format(total_file_size/1000000))
        
        # display warning for individual large files (>1MB)
        large_text = [text['metadata']['name'] for text in self.file_uploader.value.values() \
                      if text['metadata']['size']>self.large_file_size and \
                          text['metadata']['name'].endswith('.txt')]
        if len(large_text)>0:
            print('The following file(s) are larger than 1MB:', large_text)
        
        
    def check_language(self, texts: list):
        '''
        Function to check the language of the text
        
        Args:
            texts: list of uploaded texts
        '''
        # detect the number of english texts
        english_text = ['english' if detect(text)=='en' else 'non-english' for text in texts]
        english_count = Counter(english_text)
        
        # print the number of english vs non-english texts
        print('Total number of texts in English: {}'.format(english_count['english']))
        print('Total number of texts in other languages: {}'.format(english_count['non-english']))
        
        
    def extract_zip(self, zip_file):
        '''
        Load zip file
        
        Args:
            zip_file: the file containing the zipped data
        '''
        # create an input folder if not already exist
        os.makedirs('input', exist_ok=True)
        
        # read and decode the zip file
        temp = io.BytesIO(zip_file['content'])
        
        # open and extract the zip file
        with ZipFile(temp, 'r') as zip:
            # extract files
            print('Extracting {}...'.format(zip_file['metadata']['name']))
            zip.extractall('./input/')
        
        # clear up temp
        temp = None
    
    
    def load_txt(self, file) -> list:
        '''
        Load individual txt file content and return a dictionary object, 
        wrapped in a list so it can be merged with list of pervious file contents.
        
        Args:
            file: the file containing the text data
        '''
        try:
            # read the unzip text file
            with open(file) as f:
                temp = {'text_name': file.name[:-4],
                        'text': f.read()
                }
            os.remove(file)
        except:
            file = self.file_uploader.value[file]
            # read and decode uploaded text
            temp = {'text_name': file['metadata']['name'][:-4],
                    'text': codecs.decode(file['content'], encoding='utf-8', errors='replace')
            }
            
            # check for unknown characters and display warning if any
            unknown_count = temp['text'].count('�')
            if unknown_count>0:
                print('We identified {} unknown character(s) in the following text: {}'.format(unknown_count, file['metadata']['name'][:-4]))
        
        return [temp]


    def load_table(self, file) -> list:
        '''
        Load csv or xlsx file
        
        Args:
            file: the file containing the excel or csv data
        '''
        if type(file)==str:
            file = self.file_uploader.value[file]['content']

        # read the file based on the file format
        try:
            temp_df = pd.read_csv(file)
        except:
            temp_df = pd.read_excel(file)
        
        # remove file from directory
        if type(file)!=bytes:
            os.remove(file)
            
        # check if the column text and text_name present in the table, if not, skip the current spreadsheet
        if ('text' not in temp_df.columns) or ('text_name' not in temp_df.columns):
            print('File {} does not contain the required header "text" and "text_name"'.format(file['metadata']['name']))
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
        # create placeholders to store all texts and zipped file names
        all_data = []; zip_files = []
        
        # read and store the uploaded files
        files = list(self.file_uploader.value.keys())
        
        # extract zip files (if any)
        for file in files:
            if file.lower().endswith('zip'):
                self.extract_zip(self.file_uploader.value[file])
                zip_files.append(file)
        
        # remove zip files from the list
        files = list(set(files)-set(zip_files))
        
        # add extracted files to files
        for file_type in ['*.txt', '*.xlsx', '*.csv']:
            files += [file for file in Path('./input').rglob(file_type) if 'MACOSX' not in str(file)]
        
        print('Reading uploaded files...')
        print('This may take a while...')
        # process and upload files
        for file in tqdm(files):
            # process text files
            if str(file).lower().endswith('txt'):
                text_dic = self.load_txt(file)
                    
            # process xlsx or csv files
            else:
                text_dic = self.load_table(file)
            all_data.extend(text_dic)
        
        # remove files and directory once finished
        os.system('rm -r ./input')
        
        # convert them into a pandas dataframe format and add unique id
        self.text_df = pd.DataFrame.from_dict(all_data)
        self.text_df = self.hash_gen(self.text_df)
        
        # clear up all_data
        all_data = []; zip_files = []
        
        # deduplicate the text_df by text_id
        if deduplication:
            self.text_df.drop_duplicates(subset='text_id', keep='first', inplace=True)
        
        def clean_text(text):
            '''
            Function to get spaCy text

            Args:
                text: the text to be processed by spaCy
            '''
            # clean empty spaces in the text
            text = sent_tokenize(text)
            text = ' '.join(text)
            
            return text
        
        print('Pre-processing uploaded text...')
        tqdm.pandas()
        self.text_df['text'] = self.text_df['text'].progress_apply(clean_text)
        
    
    def check_mwe(self, token) -> str:
        '''
        Function to check if a token is part of multi-word expressions

        Args:
            token: the spaCy token to check
        '''
        #start_index = token._.pymusas_mwe_indexes[0][0]
        #end_index = token._.pymusas_mwe_indexes[0][1]
        return ['yes' if (token._.pymusas_mwe_indexes[0][1]-\
                         token._.pymusas_mwe_indexes[0][0])>1 else 'no'][0]
        #return ['yes: '+str(token.sent[start_index:end_index]) if (end_index-start_index)>1 else 'no'][0]
    
    
    def remove_symbols(self, text: str) -> str:
        '''
        Function to remove special symbols from USAS tags

        Args:
            text: the USAS tag to check
        '''
        text = re.sub('m','',text)
        text = re.sub('f','',text)
        text = re.sub('%','',text)
        text = re.sub('@','',text)
        text = re.sub('c','',text)
        text = re.sub('n','',text)
        text = re.sub('i','',text)
        text = re.sub(r'([+])\1+', r'\1', text)
        text = re.sub(r'([-])\1+', r'\1', text)
        
        return text
    
    
    def usas_tags_def(self, token) -> list:
        '''
        Function to interpret the definition of the USAS tag

        Args:
            token: the token containing the USAS tag to interpret
        '''
        usas_tags = token._.pymusas_tags[0].split('/')
        
        return [self.usas_tags[self.remove_symbols(usas_tag)]\
                if self.remove_symbols(usas_tag) in self.usas_tags.keys()\
                    else usas_tag\
                        for usas_tag in usas_tags]
    
    
    def token_usas_tags(self, token) -> str:
        '''
        Function to add the USAS tag to the token

        Args:
            token: the token to be added with the USAS tag 
        '''
        try:
            token_tag = token.text + ' (' + self.usas_tags_def(token)[0] + ', ' + self.usas_tags_def(token)[1] + ')'
        except:
            token_tag = token.text + ' (' + self.usas_tags_def(token)[0] + ')'
        
        return token_tag
    
    
    def highlight_sentence(self, token) -> str:
        '''
        Function to highlight selected token in the sentence

        Args:
            token: the token to be highlighted
        '''
        sentence = token.sent
        start_index = token._.pymusas_mwe_indexes[0][0]
        end_index = token._.pymusas_mwe_indexes[0][1]
        
        if end_index-start_index>1:# and end_index!=(len(sentence)-1):
            new_sentence = []
            for token in sentence:
                if token.i==start_index:
                    highlight_word_start = '<span style="color: #2ca25f; font-weight: bold">{}'.format(str(token.text))
                    new_sentence.append(highlight_word_start+token.whitespace_)
                elif token.i==end_index-1:
                    highlight_word_end = '{}</span>'.format(str(token.text))
                    new_sentence.append(highlight_word_end+token.whitespace_)
                else:
                    new_sentence.append(token.text+token.whitespace_)
            text = ''.join(new_sentence)
        else:
            word = token.text
            word_index = token.i
            highlight_word = '<span style="color: #2ca25f; font-weight: bold">{}</span>'.format(word)
            text = ''.join([token.text+token.whitespace_ \
                            if token.i!=word_index else highlight_word+token.whitespace_ \
                                for token in sentence])
        
        return text
    
    
    def add_tagger(self, 
                   text_name: str,
                   text_id:str, 
                   doc) -> pd.DataFrame:
        '''
        add semantic tags to the texts and convert into pandas dataframe

        Args:
            text_name: the text_name of the text to be tagged by the semantic tagger
            text_id: the text_id of the text to be tagged by the semantic tagger
            text: the text to be tagged by the semantic tagger
        '''
        if self.mwe=='yes':
            # extract the semantic tag for each token
            tagged_text = [{'text_name':text_name,
                            'text_id':text_id,
                            'token':token.text,
                            'pos':token.pos_,
                            'usas_tags': token._.pymusas_tags[0].split('/'),
                            'usas_tags_def': self.usas_tags_def(token),
                            'mwe': self.check_mwe(token),
                            'lemma':token.lemma_,
                            'sentence':self.highlight_sentence(token)} for token in doc]
        else:
            # extract the semantic tag for each token
            tagged_text = [{'text_name':text_name,
                            'text_id':text_id,
                            'token':token.text,
                            'pos':token.pos_,
                            'usas_tags': token._.pymusas_tags[0].split('/'),
                            'usas_tags_def': self.usas_tags_def(token),
                            'lemma':token.lemma_,
                            'sentence':self.highlight_sentence(token)} for token in doc]
        
        # convert output into pandas dataframe
        tagged_text_df = pd.DataFrame.from_dict(tagged_text)
        
        return tagged_text_df
    
    
    def tag_text(self):
        '''
        Function to iterate over uploaded texts and add semantic taggers to them
        '''
        if self.mwe=='no':
            if len(self.text_df)<500:
                n_process=1
            else:
                n_process=self.cpu_count
        else:
            n_process=1
        # iterate over texts and tag them
        for n, doc in enumerate(tqdm(self.nlp.pipe(self.text_df['text'].to_list(),
                                                n_process=n_process),
                                  total=len(self.text_df))):
            text_name = self.text_df.text_name[self.text_df.index[n]]
            text_id = self.text_df.text_id[self.text_df.index[n]]
            tagged_text = self.add_tagger(text_name, 
                                          text_id, 
                                          doc)
            self.tagged_df = pd.concat([self.tagged_df,tagged_text])
            try:
                text_name = self.text_df.text_name[self.text_df.index[n]]
                text_id = self.text_df.text_id[self.text_df.index[n]]
                tagged_text = self.add_tagger(text_name, 
                                              text_id, 
                                              doc)
                self.tagged_df = pd.concat([self.tagged_df,tagged_text])
            
            except:
                # provide warning if text is too large
                print('{} is too large. Consider breaking it \
                      down into smaller texts (< 1MB each file).'.format(text_name))
        
        # reset the pandas dataframe index after adding new tagged text
        self.tagged_df.reset_index(drop=True, inplace=True)
        
        
    def display_tag_text(self, left_right: str): 
        '''
        Function to display tagged texts 
        '''
        # widgets for selecting text_name to analyse
        enter_text, text = self.select_text_widget()
        
        # widget to analyse tags
        display_button, display_out = self.click_button_widget(desc='Display tagged text',
                                                       margin='12px 0px 0px 0px',
                                                       width='150px')
        
        # the widget for statistic output
        stat_out = widgets.Output()
        
        # widget to filter pos
        filter_pos, select_pos = self.select_multiple_options('<b>pos:</b>',
                                                              ['all'],
                                                              ['all'])
        
        # widget to filter usas_tags
        filter_usas, select_usas = self.select_multiple_options('<b>usas tag:</b>',
                                                              ['all'],
                                                              ['all'])
        
        # widget to filter mwe
        filter_mwe, select_mwe = self.select_multiple_options('<b>mwe:</b>',
                                                              ['all','yes','no'],
                                                              ['all'])
        
        # function to define what happens when the button is clicked
        def on_display_button_clicked(_):
            # display selected tagged text
            with display_out:
                clear_output()
                
                # get selected text
                self.text_name[left_right] = text.value
                
                # display the selected text
                self.df[left_right] = self.tagged_df[self.tagged_df['text_name']==self.text_name[left_right]].iloc[:,2:].reset_index(drop=True)
                
                # for new selected text
                if self.text_name[left_right]!=self.selected_text[left_right]:
                    self.selected_text[left_right]=self.text_name[left_right]
                    
                    # generate usas tag options
                    usas_list = self.df[left_right].usas_tags_def.to_list()
                    usas_list = [item for sublist in usas_list for item in sublist]
                    usas_list = sorted(list(set(usas_list)))
                    usas_list.insert(0,'all')
                    select_usas.options = usas_list
                    
                    # generate pos options
                    new_pos = sorted(list(set(self.df[left_right].pos.to_list())))
                    new_pos.insert(0,'all')
                    select_pos.options = new_pos
                    
                    select_pos.value=('all',)
                    select_usas.value=('all',)
                    select_mwe.value=('all',)
                
                # get the filter values
                inc_pos = select_pos.value
                inc_usas = select_usas.value
                inc_mwe = select_mwe.value
                
                # display based on selected filter values
                if inc_usas!=('all',):
                    usas_index=[]
                    for selected_usas in inc_usas:
                        index = [n for n, item in enumerate(self.df[left_right].usas_tags_def.to_list()) \
                                 if selected_usas in item]
                        usas_index.extend(index)
                    usas_index = list(set(usas_index))
                    self.df[left_right] = self.df[left_right].iloc[usas_index]
                
                if inc_pos!=('all',):
                    self.df[left_right] = self.df[left_right][self.df[left_right]['pos'].isin(inc_pos)]
                
                if inc_mwe!=('all',):
                    self.df[left_right] = self.df[left_right][self.df[left_right]['mwe'].isin(inc_mwe)]
                
                pd.set_option('display.max_rows', None)
                
                print('Tagged text: {}'.format(self.text_name[left_right]))
                # display in html format for styling purpose
                df_html = self.df[left_right].to_html(escape=False)
                
                # Concatenating to single string
                df_html = self.style+'<div class="dataframe-div">'+df_html+"\n</div>"
                
                #display(HTML(all_html))
                display(HTML(df_html))
                
            with stat_out:
                clear_output()
                print('Statistical information: {}'.format(self.text_name[left_right]))
                count_usas = pd.DataFrame.from_dict(Counter(list(itertools.chain(*self.df[left_right].usas_tags_def.to_list()))),
                                                    orient='index', columns=['usas_tag']).T
                count_pos = pd.DataFrame.from_dict(Counter(self.df[left_right].pos.to_list()),
                                                   orient='index', columns=['pos']).T
                count_mwe = pd.DataFrame.from_dict(Counter(self.df[left_right].mwe.to_list()),
                                                   orient='index', columns=['mwe']).T
                
                count_all = pd.concat([count_usas, count_pos, count_mwe])
                count_all = count_all.fillna('-')
                all_html = count_all.to_html(escape=False)
                all_html = self.style+'<div class="dataframe-div">'+all_html+"\n</div>"
                display(HTML(all_html))
                
                #pd.options.display.max_colwidth = 50
        
        # link the button with the function
        display_button.on_click(on_display_button_clicked)
        
        hbox1 = widgets.HBox([enter_text, text],
                             layout = widgets.Layout(height='35px'))
        hbox2 = widgets.HBox([filter_pos, select_pos],
                             layout = widgets.Layout(width='250px',
                                                     margin='0px 0px 0px 43px'))
        hbox3 = widgets.HBox([filter_usas, select_usas],
                             layout = widgets.Layout(width='300px',
                                                     margin='0px 0px 0px 13px'))
        if self.mwe=='yes':
            hbox4 = widgets.HBox([filter_mwe, select_mwe],
                                 layout = widgets.Layout(width='300px',
                                                         margin='0px 0px 0px 36px'))
            hbox5a = widgets.HBox([hbox2, hbox3])
            hbox5 = widgets.VBox([hbox5a, hbox4])
        else:
            hbox5 = widgets.HBox([hbox2, hbox3])
        hbox6 = widgets.HBox([display_button],
                             layout=Layout(margin= '0px 0px 10px 75px'))
        #vbox = widgets.VBox([hbox1, hbox5, hbox6, display_out],
        #                     layout = widgets.Layout(width='500px'))
        vbox = widgets.VBox([hbox1, hbox5, hbox6],
                             layout = widgets.Layout(width='500px'))
        
        return vbox, display_out, stat_out
    
    
    def display_two_tag_texts(self): 
        '''
        Function to display tagged texts commparison 
        '''
        # widget for displaying first text
        vbox1, display_out1, stat_out1 = self.display_tag_text('left')
        vbox2, display_out2, stat_out2 = self.display_tag_text('right')
        
        hbox = widgets.HBox([vbox1, vbox2])
        vbox = widgets.VBox([hbox, display_out1, display_out2, stat_out1, stat_out2])
        
        return vbox
        
        
    def save_tag_text(self, 
                      start_index: int, 
                      end_index: int):
        '''
        Function to save tagged texts into an excel spreadsheet using text_name as the sheet name

        Args:
            start_index: the start index of the text to be saved
            end_index: the end index of the text to be saved
        '''
        # define the file_name
        file_name = 'tagged_text_{}_to_{}.xlsx'.format(start_index, min(end_index,len(self.text_df)))
        
        # open an excel workbook
        wb = Workbook()
        
        # empty variables to check duplicate name sheets
        sheet_names = []; n=0
        
        # tag texts and save to new sheets in the excel spreadsheet
        for text in tqdm(self.text_df[start_index:end_index].itertuples(), total=len(self.text_df[start_index:end_index])):
            try:
                tagged_text = self.tagged_df[self.tagged_df['text_id']==text.text_id].iloc[:,2:]
                sheet_name = text.text_name[:10]
                if sheet_name in sheet_names:
                    sheet_name += str(n)
                    n+=1
                sheet_names.append(sheet_name)
                values = [tagged_text.columns] + list(tagged_text.values)
                wb.new_sheet(sheet_name, data=values)
            except:
                print('{} is too large. Consider breaking it down into smaller texts (< 1MB each file).'.format(text.text_name))
        
        # save the excel spreadsheet
        wb.save(file_name)
        print('Semantic tags successfully added and saved into {}!'.format(file_name))
    
    
    def top_entities(self, 
                     count_ent: dict, 
                     top_n: int) -> dict:
        '''
        Function to identify top entities in the text
        
        Args:
            count_ent: the count of the selected entity
            top_n: the number of top items to be shown
        '''
        # count and identify top entity
        top_ent = dict(sorted(count_ent.items(), key=lambda x: x[1], reverse=False)[-top_n:])
        
        return top_ent
        
        
    def count_entities(self, which_text: str, which_ent: str) -> dict:
        '''
        Function to count the number of selected entities in the text
        
        Args:
            which_ent: the selected entity to be counted
        '''
        if which_text=='all texts':
            df = self.tagged_df
        else:
            df = self.tagged_df[self.tagged_df['text_name']==which_text].reset_index(drop=True)
        
        # exclude punctuations
        items_to_exclude = ['PUNCT', ['PUNCT']]
        
        import itertools

        number_of_cpu = joblib.cpu_count()        
        # count entities based on type of entities
        if which_ent=='usas_tags' or which_ent=='usas_tags_def':
            # identify usas_tags or usas_tags_def
            print('Analysing text...')
            ent = [item for item in tqdm(list(itertools.chain(*df[which_ent].to_list()))) \
                   if item not in items_to_exclude]
            
        elif which_ent=='lemma' or which_ent=='pos' or which_ent=='token':
            # identify lemmas, tokens or pos tags
            ent = [item for n, item in enumerate(df[which_ent].to_list()) 
                   if df['pos'][n] not in items_to_exclude]
            
        elif which_ent=='mwe':
            # identify mwe indexes
            all_mwe = set(zip(df[df['mwe']=='yes']['start_index'],\
                              df[df['mwe']=='yes']['end_index']))
            
            # join the mwe expressions
            ent = [' '.join([self.tagged_df.loc[i,'token'] \
                             for i in range(mwe[0],mwe[1])]) for mwe in all_mwe]
        
        return Counter(ent)
    
    
    def count_text(self, which_text: str, which_ent, inc_ent):
        '''
        Function to identify texts based on selected top entities
        
        Args:
            which_ent: the selected entity, e.g., USAS tags, POS tags, etc.
            inc_ent: the included entity type, e.g., Z1, VERB, etc.
        '''
        if which_text=='all texts':
            df = self.tagged_df
        else:
            df = self.tagged_df[self.tagged_df['text_name']==which_text].reset_index(drop=True)
            
        # placeholder for selected texts
        selected_texts = []
        
        # iterate over selected entities and identified tokens based on entity type
        for n, tag in enumerate(df[which_ent]):
            if type(tag)==list:
                for i in tag:
                    if i in inc_ent:
                        selected_texts.append(df['token'][n])
            else:
                if tag in inc_ent:
                    selected_texts.append(df['token'][n])
                    
        return Counter(selected_texts)
    
    
    def visualize_stats(
            self, 
            which_text: str,
            top_ent: dict,
            sum_ent: int,
            top_n: int,
            title: str,
            color: str
            ):
        '''
        Create a horizontal bar plot for displaying top n named entities
        
        Args:
            top_ent: the top entities to display
            top_ent: the number of top entities to display
            title: title of the bar plot
            color: color of the bars
        '''
        if top_ent!={}:
            # specify the width, height and tick range for the plot
            display_height = top_n/2
            range_tick = max(1,round(max(top_ent.values())/5))
            
            # visualize the entities using horizontal bar plot
            fig = plt.figure(figsize=(10, max(5,display_height)))
            plt.barh(list(top_ent.keys()), 
                     list(top_ent.values()),
                     color=color)
            
            # display the values on the bars
            for i, v in enumerate(list(top_ent.values())):
                plt.text(v+(len(str(v))*0.05), 
                         i, 
                         '  {} ({:.0f}%)'.format(v,100*v/sum_ent), 
                         fontsize=10)
            
            # specify xticks, yticks and title
            plt.xticks(range(0, max(top_ent.values())+int(2*range_tick), 
                             range_tick), 
                       fontsize=10)
            plt.yticks(fontsize=10)
            bar_title = 'Top {} "{}" in text: "{}"'.format(min(top_n,
                                                           len(top_ent.keys())),
                                                             title, 
                                                             which_text)
            plt.title(bar_title, fontsize=12)
            plt.show()
            
        return fig, bar_title
        
        
    def analyse_tags(self):
        '''
        Function to display options for analysing entity/tag
        '''
        # options for bar chart titles
        titles = {'usas_tags': 'USAS tags',
                  'usas_tags_def': 'USAS tag definitions',
                  'pos': 'Part-of-Speech Tags',
                  'lemma': 'lemmas',
                  'token': 'tokens'}
        
        # entity options
        ent_options = ['usas_tags_def', #'usas_tags', 
                       'pos', 'lemma', 'token']
        
        if self.mwe=='yes':
            titles['mwe']='Multi-Word Expressions'
            ent_options.append('mwe')
        
        # placeholder for saving bar charts
        self.figs = []
        
        # widgets for selecting text_name to analyse
        choose_text, my_text = self.select_text_widget(entity=True)
        
        # widget to select entity options
        enter_entity, select_entity = self.select_options('<b>Select entity to show:</b>',
                                                        ent_options,
                                                        'usas_tags_def')
        
        # widget to select n
        enter_n, top_n = self.select_n_widget('<b>Select n:</b>', 5)
        
        # widget to analyse tags
        analyse_button, analyse_out = self.click_button_widget(desc='Show top entities',
                                                       margin='20px 0px 0px 0px',
                                                       width='155px')
        
        # function to define what happens when the button is clicked
        def on_analyse_button_clicked(_):
            # clear save_out
            with save_out:
                clear_output()
                
            # clear analyse_top_out
            with analyse_top_out:
                clear_output()
                
            # display bar chart for selected entity
            with analyse_out:
                clear_output()
                
                # get selected values
                which_text=my_text.value
                which_ent=select_entity.value
                n=top_n.value
                title=titles[which_ent]
                
                # get top entities
                count_ent = self.count_entities(which_text, which_ent)
                sum_ent = sum(count_ent.values())
                top_ent = self.top_entities(count_ent, n)
                
                # create bar chart
                fig, bar_title = self.visualize_stats(which_text,
                                                      top_ent,
                                                      sum_ent,
                                                      n,
                                                      title,
                                                      '#2eb82e')
                
                # append to self.figs for saving later
                self.figs.append([fig, bar_title])
                
                # update options for displaying tokens in entity type
                if which_ent!='mwe' and which_ent!='token':
                #if which_ent!='token':
                    new_options = list(top_ent.keys())
                    new_options.reverse()
                    select_text.options = new_options
                    select_text.value = [new_options[0]]
                    enter_text.value = '<b>Select {}:</b>'.format(titles[which_ent][:-1])
                else:
                    select_text.options = ['None']
                    select_text.value = ['None']
                    enter_text.value = '<b>No selection required.</b>'
        
        # link the button with the function
        analyse_button.on_click(on_analyse_button_clicked)
        
        # widget to select top entity type and display top tokens
        enter_text, select_text = self.select_multiple_options('<b>Select tag/lemma/token:</b>',
                                                               ['None'],
                                                               ['None'])
        
        # widget to select n
        enter_n_text, top_n_text = self.select_n_widget('<b>Select n:</b>', 5)
        
        # widget to analyse texts
        analyse_top_button, analyse_top_out = self.click_button_widget(desc='Show top words', 
                                                                       margin='12px 0px 0px 0px',
                                                                       width='155px')
        
        # function to define what happens when the button is clicked
        def on_analyse_top_button_clicked(_):
            # clear save_out
            with save_out:
                clear_output()
            
            # display bar chart for selected entity type
            with analyse_top_out:
                # obtain selected entity type
                which_ent=select_entity.value
                
                # only create new bar chart if not 'mwe' or 'token' (already displayed)
                if which_ent!='mwe' and which_ent!='token':
                #if which_ent!='token':
                    # get selected values
                    which_text=my_text.value
                    clear_output()
                    inc_ent=select_text.value
                    n=top_n_text.value
                    
                    # display bar chart for every selected entity type
                    for inc_ent_item in inc_ent:
                        title = inc_ent_item
                        count_ent = self.count_text(which_text, which_ent, inc_ent_item)
                        sum_ent = sum(count_ent.values())
                        top_text = self.top_entities(count_ent, n)
                        
                        try:
                            fig, bar_title = self.visualize_stats(which_text,
                                                                  top_text,
                                                                  sum_ent,
                                                                  n,
                                                                  title,
                                                                  '#008ae6')
                            self.figs.append([fig, bar_title])
                        except:
                            print('Please show top entities first!')
                else:
                    # display warning for 'mwe' or 'token'
                    with analyse_out:
                        print('The top {} are shown above!'.format(titles[which_ent]))
        
        # link the button with the function
        analyse_top_button.on_click(on_analyse_top_button_clicked)
        
        # widget to save the above
        save_button, save_out = self.click_button_widget(desc='Save analysis', 
                                                         margin='10px 0px 0px 0px',
                                                         width='155px')
        
        # function to define what happens when the save button is clicked
        def on_save_button_clicked(_):
            with save_out:
                clear_output()
                if self.figs!=[]:
                    # set the output folder for saving
                    out_dir='./output/'
                    
                    print('Analysis saved! Click below to download:')
                    # save the bar charts as jpg files
                    for fig, bar_title in self.figs:
                        file_name = '-'.join(bar_title.split()) + '.jpg'
                        fig.savefig(out_dir+file_name, bbox_inches='tight')
                        display(DownloadFileLink(out_dir+file_name, file_name))
                    
                    # reset placeholder for saving bar charts
                    self.figs = []
                else:
                    print('You need to generate the bar charts before you can save them!')
        
        # link the save_button with the function
        save_button.on_click(on_save_button_clicked)
        
        # displaying inputs, buttons and their outputs
        hbox1 = widgets.HBox([choose_text, my_text],
                             layout = widgets.Layout(height='35px'))
        vbox1 = widgets.VBox([enter_entity,
                              select_entity,
                              enter_n, top_n,], 
                             layout = widgets.Layout(width='250px', height='151px'))
        vbox2 = widgets.VBox([analyse_button,
                              save_button], 
                             layout = widgets.Layout(width='250px', height='100px'))
        vbox3 = widgets.VBox([vbox1, vbox2])
        vbox4 = widgets.VBox([enter_text, 
                              select_text, 
                              enter_n_text, top_n_text,
                              analyse_top_button],
                             layout = widgets.Layout(width='250px', height='250px'))
        
        hbox2 = widgets.HBox([vbox3, vbox4])
        vbox = widgets.VBox([hbox1, hbox2, save_out, analyse_out, analyse_top_out],
                            layout = widgets.Layout(width='500px'))
        
        return vbox
    
    
    def analyse_two_tags(self):
        '''
        Function to display options for comparing text analysis
        '''
        vbox1 = self.analyse_tags()
        vbox2 = self.analyse_tags()
        
        hbox = widgets.HBox([vbox1, vbox2])
        
        return hbox
        
    
    def save_options(self):
        '''
        options for saving tagged texts
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value='<b>Select the tagged texts to save (up to {} texts at a time):</b>'.format(self.max_to_process),
            placeholder='',
            description=''
            )
        
        # widgets for selecting the number of texts to process in each batch
        enter_start_n, start_n = self.select_n_widget('Start index:', 0)
        enter_end_n, end_n = self.select_n_widget('End index:', 50)
        
        # the output after clicking the button
        text_out = widgets.Output()
        
        with text_out:
            batch_size = end_n.value - start_n.value
            pd.set_option('display.max_rows', self.max_to_process)
            
            # display texts to be saved
            display(self.text_df[start_n.value:end_n.value])
            
        # give notification when file is uploaded
        def _cb(change):
            with text_out:
                clear_output()
                if (end_n.value-start_n.value)>self.max_to_process:
                    print('You can select only up to 50 texts. Please revise the start/end index.')
                else:
                    pd.set_option('display.max_rows', self.max_to_process)
                    display(self.text_df[start_n.value:end_n.value])
            
        # observe when file is uploaded and display output
        start_n.observe(_cb, names='value')
        end_n.observe(_cb, names='value')
        
        # widget to process texts
        process_button, process_out = self.click_button_widget(desc='Save tagged texts', 
                                                       margin='10px 0px 0px 0px',
                                                       width='160px')
        
        # function to define what happens when the top button is clicked
        def on_process_button_clicked(_):
            with process_out:
                clear_output()
                
                if (end_n.value-start_n.value)<=self.max_to_process:
                    # process selected texts
                    self.save_tag_text(start_n.value, end_n.value)
                    print('text index {} to {} have been processed. Click below to download:'.format(start_n.value, end_n.value))
                    
                    # download the excel spreadsheet onto your computer
                    file_name = 'tagged_text_{}_to_{}.xlsx'.format(start_n.value, min(end_n.value,len(self.text_df)))
                    display(DownloadFileLink(file_name, file_name))
                    
                    # change index values to save the next batch
                    start_n.value = end_n.value+1
                    end_n.value = min((start_n.value + self.max_to_process),
                                      (len(self.text_df)-(end_n.value-start_n.value)-1))
                else:
                    print('You can select only up to {} texts. Please revise the start/end index.'.format(self.max_to_process))
                    
        # link the top_button with the function
        process_button.on_click(on_process_button_clicked)
        
        # displaying inputs, buttons and their outputs
        vbox1 = widgets.VBox([enter_text, 
                              enter_start_n, start_n,
                              enter_end_n, end_n], 
                             layout = widgets.Layout(width='600px', height='170px'))
        vbox2 = widgets.VBox([process_button, process_out],
                             layout = widgets.Layout(width='600px'))#, height='80px'))
        
        vbox = widgets.VBox([vbox1, vbox2, text_out])
        
        return vbox
    
    
    def select_text_widget(self, entity: bool=False):
        '''
        Create widgets for selecting text to display
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value='<b>Select text:</b>',
            placeholder='',
            description=''
            )
        
        # use text_name for text_options
        text_options = self.text_df.text_name.to_list() # get the list of text_names
        
        # the option to select 'all texts' for analysing top entities
        if entity:
            text_options.insert(0, 'all texts')
        
        # widget to display text_options
        text = widgets.Combobox(
            placeholder='Choose text to display...',
            options=text_options,
            description='',
            ensure_option=True,
            disabled=False,
            layout = widgets.Layout(width='195px')
        )
        
        return enter_text, text
    
    
    def select_options(self, 
                       instruction: str,
                       options: list,
                       value: str):
        '''
        Create widgets for selecting the number of entities to display
        
        Args:
            instruction: text instruction for user
            options: list of options for user
            value: initial value of the widget
        '''
        # widget to display instruction
        enter_text = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widget to select entity options
        select_option = widgets.Dropdown(
            options=options,
            value=value,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
            )
        
        return enter_text, select_option
    
    
    def select_multiple_options(self, 
                                instruction: str,
                                options: list,
                                value: list):
        '''
        Create widgets for selecting muyltiple options
        
        Args:
            instruction: text instruction for user
            options: list of options for user
            value: initial value of the widget
        '''
        # widget to display instruction
        enter_m_text = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widget to select entity options
        select_m_option = widgets.SelectMultiple(
            options=options,
            value=value,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
            )
        
        return enter_m_text, select_m_option
        
        
    def select_n_widget(self, 
                        instruction: str, 
                        value: int):
        '''
        Create widgets for selecting the number of entities to display
        
        Args:
            instruction: text instruction for user
            value: initial value of the widget
        '''
        # widget to display instruction
        enter_n = widgets.HTML(
            value=instruction,
            placeholder='',
            description=''
            )
        
        # widgets for selecting n
        n_option = widgets.BoundedIntText(
            value=value,
            min=0,
            #max=len(self.text_df),
            step=5,
            description='',
            disabled=False,
            layout = widgets.Layout(width='150px')
        )
        
        return enter_n, n_option
    
    
    def click_button_widget(
            self, 
            desc: str, 
            margin: str='10px 0px 0px 10px',
            width='320px'
            ):
        '''
        Create a widget to show the button to click
        
        Args:
            desc: description to display on the button widget
            margin: top, right, bottom and left margins for the button widget
            width: the width of the button
        '''
        # widget to show the button to click
        button = widgets.Button(description=desc, 
                                layout=Layout(margin=margin, width=width),
                                style=dict(font_weight='bold'))
        
        # the output after clicking the button
        out = widgets.Output()
        
        return button, out