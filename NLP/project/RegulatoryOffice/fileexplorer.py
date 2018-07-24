# encoding: utf-8

"""
Retrieving data from legal docs on special positions...

KnowledgeGraph
    - Creates a json dump where we can store all the info on the EURLEX files

DocumentFileIndexer
    - Used to index an EURLEX html file, save the indices of the text for later extraction

ExporeRegDoc
    - Used to explore certain parts of the EURLEX html files, is going to be removed soon

ExtractRegDoc
    - Used to extract text from an EURLEX document (Child of ExploreRegDoc). Will be
    using the results from DOcumentFileIndexer to save the results in a better format

"""

import os
import itertools
import re
import string
from helper.miscfunction import n_diff_cluster, get_item_occurence
from deprecated import deprecated

import json
import glob


class KnowledgeGraph:
    """
    Here we want to initiate or build upon an existing json file that contains information about EURLEX documents.

    Not sure yet if the key should be the eurlex ID or the eurlex file name (hence + extention)
    """

    def __init__(self, db_src_path, name='file_analysis.json', dir_file=None, file_ext=None, iscelex=False):
        """

        :param db_src_path: location of the db that we are going to store as a json...
        :param name: name of the json dump file
        :param dir_file: list of input files that need to be processed (result from glob.glob(\\*.html) Hence full path
        :param file_ext: used to clean the name of dir_file by specifying the file extention
        :param iscelex: optional parameter to include specific cleaning options when dealing with celex documents.
        """

        self.db_src_path = db_src_path
        self.db_name = name
        self.db_full_path = os.path.join(self.db_src_path, self.db_name)

        if os.path.isfile(self.db_full_path):
            if os.path.getsize(self.db_full_path) == 0:
                self._init_json(dir_file, file_ext, iscelex)
            else:
                self.db_json = self.load_db_json()
        else:
            self._init_json(dir_file, file_ext, iscelex)

    def _init_json(self, dir_file, file_ext, iscelex):
        """
        Initiates a first dict..
        :return:
        """
        assert dir_file is not None
        assert file_ext is not None

        self.n_file = len(dir_file)
        self.n_ext = len(file_ext)
        self.dir_file = dir_file
        self.file_name_ext = [os.path.basename(x) for x in dir_file]
        self.file_name = [x[:-self.n_ext] for x in self.file_name_ext if x.endswith(file_ext)]

        if iscelex:
            # TODO Could remove the CELEX prefix... or I could just store the files without CELEX prefix
            pass

        if not os.path.isfile(self.db_full_path) or os.path.getsize(self.db_full_path) == 0:
                std_dict = {self.file_name[i]: {'Raw location': self.dir_file[i]} for i in range(self.n_file)}
                with open(self.db_full_path, 'w') as f:
                    json.dump(std_dict, f)

        self.db_json = self.load_db_json()

    def load_db_json(self):
        """
        This should be put into the file managing stuff I guess?
        :return:
        """
        with open(self.db_full_path) as f:
            return json.load(f)

    def write_db_json(self):
        """
        This should be put into the file managing stuff I guess?
        :return:
        """
        with open(self.db_full_path, 'w') as f:
            json.dump(self.db_json, f)

    def update_json(self, data):
        """
        Test
        :param data:
        :return:
        """
        assert isinstance(data, dict)
        assert all([isinstance(v, dict) for k, v in data.items()])

        for k_filename, v_propdict in data.items():
            for k_propname, v_propvalue in v_propdict.items():
                if k_propname in self.db_json[k_filename].keys():
                    self.db_json[k_filename][k_propname] = v_propvalue
                    print('updated a value...', k_filename, k_propname, v_propvalue)
                else:
                    self.db_json[k_filename][k_propname] = v_propvalue
                    print('implemented a value...', k_filename, k_propname, v_propvalue)

    def search(self, query):
        """
        Having a starting search funcitonality now.. could be better..

        Search job on the loaded dict..
        :param query:
        :return:
        """

        keyword_things = []
        for k, v in self.db_json.items():
            for i in v.values():
                if isinstance(i, list):
                    for j in i:
                        if re.search(query, j, re.IGNORECASE):
                            keyword_things.append(k)
                elif isinstance(i, str):
                    # If we find a string value... treat it like one
                    if re.search(query, i, re.IGNORECASE):
                        keyword_things.append(k)
                else:
                    # If we find an integer of some sort.. convert it to str..
                    if re.search(query, str(i), re.IGNORECASE):
                        keyword_things.append(k)
        return list(set(keyword_things))



class DocumentFileIndexer:
    """
    TODO under construction...

    Let's see if we can find a way to index documents.. and then use these indices in next classes.

    This is a big preprocessing step for the reuglatory documents. And can be a nice train/test case for other (
    unstructured) documents)
    """

    def __init__(self, input_html, name):
        self.input_html = input_html
        self.name = name
        self.classes = [element['class'][0] for element in self.input_html.find_all(class_=True)]
        self.index_text = [(i, x['class'][0], x.text) for i, x in enumerate(input_html.find_all(class_=True))]
        self.titel_classes = [x for x in set(self.classes) if re.findall('ti', x)]

    def index_titles(self):
        """
        Returns a list of tuples with (index, title-class, header, title text)

        :return:
        """
        # Find all title classes inside this document

        title_text = sorted([x for x in self.index_text if x[1] in self.titel_classes])

        index_title, type_title, content_title = zip(*title_text)
        index_cluster = list(n_diff_cluster(index_title, 1, return_index=True))
        title_text_concat = [(index_title[i_cluster[0]], type_title[i_cluster[0]], content_title[i_cluster[0]],
                              ' '.join([content_title[i] for i in i_cluster[1::]]).replace('\n', '')) for i_cluster in
                             index_cluster]
        return title_text_concat

    def index_articles(self):
        """

        :return:
        """

        pass

    def index_references(self):
        """
        maybe these can also be usefull.. not sure..
        :return:
        """

        pass


class ExploreRegDoc:
    """
    Used to explore html docs obtained through EurLEX
    """

    def __init__(self, input_html, name, rid_data):
        self.input_html = input_html
        self.name = name

        self.input_class_true = self.input_html.find_all(class_=True)
        self.input_class_false = self.input_html.find_all(class_=False)

        self.text_class_true = [(i, x['class'][0], x.text) for i, x in enumerate(self.input_class_true)]
        self.text_class_false = [(i, x.name, x.text) for i, x in enumerate(self.input_class_false)]

        self.classes = [element['class'][0] for element in self.input_html.find_all(class_=True)]
        dum = [i for i, x in enumerate(rid_data['Hyperlink to Regulatory Document']) if re.findall(
            self.name.replace('_', ':'), x)]
        if len(dum) == 1:
            self.id_html = dum[0]
        else:
            print('We have found the doc multiple times in RID, index: {0}. Using first option as reference.'.format(
                dum))
            self.id_html = dum[0]

        self.rid_data = {key: value[self.id_html] for key, value in rid_data.items()}

    @deprecated('This function will soon move to ...')
    def _get_department(self):
        """
        Use the RID database to check whether we can find the CELEX nr
        :return:
        """
        # 'Regulatory Topic'  # can also be fun
        # 'Regulation Type'  # - drafts.. regulations..etc..
        return self.rid_data['Link to Existing Regulatory File']

    @deprecated('This function will soon move to ...')
    def _get_impact(self):
        """
        :return: tuple (size 3) with the impact in string
        """
        impact_function = self.rid_data['Impact on Functions (2nd LOD)']
        impact_business = self.rid_data['Impact on Business (1st LOD)']
        impact_country = self.rid_data['Impact on Countries']

        return impact_function, impact_business, impact_country

    @deprecated('This function will soon move to ...')
    def _get_topic_type(self):
        """
        Returns topic/type of the document
        :return:
        """
        reg_topic = self.rid_data['Regulatory Topic']
        reg_type = self.rid_data['Regulation Type']

        return reg_topic, reg_type

    @deprecated('This function will soon move to ...')
    def get_properties(self):
        """
        Return the classes inside input_html..

        :return: a set with all the classes
        """
        properties = [['name', 'class', 'content', 'doc-ti', 'ti-art', 'sti-art', 'hd-ti']]
        for element in self.input_class_true:
            temp = [None] * 7
            temp[0] = self.name
            temp[1] = element['class'][0]  # We need to actually say that it is a new list to copy it
            temp[2] = element.text
            # temp[3] = get_item_occurence()

            n = 2
            if 'doc-ti' in element['class']:
                temp[n+1] = element.text
            elif 'ti-art' in element['class']:
                temp[n+2] = element.text
            elif 'sti-art' in element['class']:
                temp[n+3] = element.text
            elif 'hd-ti' in element['class']:
                temp[n+4] = element.text

            properties.append(temp)

        return properties

    @staticmethod
    def transform_to_celex(input_list, doc_type):
        """
        This method returns a CELEX string from an input date in the format YYYY/II/SS

        Sector 3 - Legislation
        L for Directives
        R for Regulations
        D for Decisions

        Sector 6 - Case-law
        CJ for Judgments by Court of Justice
        CC for Opinions of the advocate-general
        CO for Orders of the Court of Justice.

        Sector 5 - Preparatory documents
        PC for Legislative proposals by the Commission (COM documents), etc.
        DC for other COM documents (green and white papers, communications, reports...)
        SC for SWD documents (staff working documents, impact assessments...)
        JC for JOIN documents (adopted jointly by the Commission and the High Representative)

        :param doc_type:
        :param input_list:
        :return:
        """
        if isinstance(input_list, str):
            input_list = list(input_list)

        output_list = []
        for i_string in input_list:
            string_search = re.search('([0-9]+)/([0-9]+)', i_string)
            year = string_search.group(1)
            if len(year) == 2:
                year = '19' + year
            doc_nr = string_search.group(2).zfill(4)

            doc_code = ''
            if doc_type in ['Directive', 'Directives']:
                doc_code = 'L'
            elif doc_type in ['Regulation', 'Regulations']:
                doc_code = 'R'
            elif doc_type in ['Decision', 'Decisions']:
                doc_code = 'D'
            elif doc_type in ['Legislative proposal', 'Legislative proposals']:
                doc_code = 'PC'
            elif doc_type in ['white paper', 'COM']:
                doc_code = 'DC'

            sector = ''
            if doc_code in ['L', 'R', 'D']:
                sector = '3'
            elif doc_code in ['PC', 'DC', 'SC', 'JC']:
                sector = '5'

            celex_id = sector + year + doc_code + doc_nr
            eurlex_html = r'http://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=celex%3A' + celex_id
            output_list.append((eurlex_html, celex_id))

        return output_list


class ExtractRegDoc(ExploreRegDoc):
    """
    Used to extract text from html documents creatd by EurLEX
    """

    def __init__(self, input_html, name, rid_data):
        super(ExtractRegDoc, self).__init__(input_html, name, rid_data)

    @deprecated
    def get_item_occurence(input_list):
        """
        Counts the occurence of subsequence elements while preserving the order in which they happened

        e.g.
        x = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'c']
        y = get_item_occurence(x)
        print(y)
            [('a', 4), ('b', 3), ('a', 2), ('c', 1)]
        :param input_list: input list with string items
        :return: list with tuples
        """
        i_count = 1
        n_class = 0
        count_classes = []
        group_classes = [n_class]
        temp_classes = input_list[0]

        for i_class in input_list[1:] + ['stop']:
            if i_class == temp_classes:
                group_classes.append(n_class)
                i_count += 1
            else:
                n_class += 1
                group_classes.append(n_class)
                count_classes.append((temp_classes, i_count))
                i_count = 1

                temp_classes = i_class

        return count_classes, group_classes[:-1]  # Second last because of the 'stop' trigger

    def get_first_block(self):
        """
        Here we get the first block of text.. right before all the articles begin.

        To be more precise.. it gets all the data from the first doc-ti until the mentioning of the l

        Underconstruction

        TODO Need to unit test this. You can do this with the red block text..
        TODO DO all the documents posses an Article 1? And what do we do when we cant find it
        TODO can also use a FileIndex class that creates indices for the location of all the special parts..
        """

        structure_count, global_group_id = get_item_occurence(self.classes)
        titel_structure, titel_count = zip(*structure_count)

        index_doc_ti = 0  # Default value for when
        if 'doc-ti' in titel_structure:
            index_doc_ti = titel_structure.index('doc-ti')  # This function always gets the first occurence.

        index_doc_ti = sum(titel_count[0:(index_doc_ti+1)])

        test = [(element['class'][0], element.text) for element in self.input_class_true]
        index_article = [derp for derp, i in enumerate(test) if i[0] == 'ti-art' and i[1] == 'Article 1'][0] - 1

        return self.input_class_true[index_doc_ti:index_article]

    def get_last_block(self):
        """

        :return:
        """
        test = [(element['class'][0], element.text) for element in self.input_class_true]
        index_article = [derp for derp, i in enumerate(test) if i[0] == 'ti-art' and i[1] == 'Article 1'][0] - 1

        return self.input_class_true[index_article::]

    def get_title_structure(self):
        """
        Returns the structure of a document in an ordered fashion

        :return: list of titel structure
        """
        title_classes = [x for x in set(self.classes) if re.findall('ti', x)]
        title_text = sorted([self.get_text_by_class(i_tit) for i_tit in title_classes])
        title_text = sorted(list(itertools.chain(*title_text)))

        index_title, type_title, content_title = zip(*title_text)
        index_cluster = list(n_diff_cluster(index_title, 1, return_index=True))
        title_text_concat = [(index_title[i_cluster[0]], type_title[i_cluster[0]], ' '.join([content_title[i] for i
                                                                                             in i_cluster]).replace(
            '\n', '')) for i_cluster in index_cluster]
        # PEP-8 formats it really weird here..
        return title_text_concat

    def get_reference_text(self, input_title, input_pointer=None):
        """
        Returns the piece of text that is associated with a certain pointer

        For example, point (a) and (b) in Article 1 of ...

        :param input_title:
        :param input_pointer: list of pointers...
        :return: a dict with the article text and the pointers..
        """
        title_struct = self.get_title_structure()
        title_range = [(x[0], title_struct[i+1][0]) for i, x in enumerate(title_struct) if input_title in x[2]]
        print(title_range)  # This one should have only length of one.. but I am not sure yet.
        derp = title_range[0]
        title_content = self.input_class_true[derp[0]:derp[1]]

        if input_pointer:
            content_pointer = []
            for i_pointer in input_pointer:
                temp = [(i_pointer, title_content[i+1].text) for i, x in enumerate(title_content) if i_pointer == x.text]
                content_pointer.append(temp)
            return content_pointer
        else:
            return [x.text for x in title_content]

    def get_reference_directives(self):
        """
        Here we are able to return the references to the directives that are present in the documents

        :return: list of references to directives and their date and whatnot...
        """
        directive_text = self.get_text_by_keyword('Directive')
        directive_text_split = [x.split() for x in directive_text]
        found_directives = [y[i+1] for y in directive_text_split for i, x in enumerate(y)
                            if re.findall('Directive', x) and (i+1) < len(y)]

        src_str = '([0-9]+/[0-9]+/[A-Z]{2})'
        extracted_dates = list(set([re.search(src_str, x).group() for x in found_directives if re.search(src_str, x)]))
        return extracted_dates

    def get_reference_regulation(self):
        """
        Here we are able to return the references to the Regulations that are present in the documents

        :return: list of references to directives and their date and whatnot...
        """
        _trans_punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))

        regulation_text = self.get_text_by_keyword('Regulation')
        regulation_text_split = [x.split() for x in regulation_text]

        found_regulations = [' '.join(y[(i+3):(i+4)]) for y in regulation_text_split for i, x in enumerate(y) if
                             re.findall('Regulation', x) and (i+1) < len(y)]

        extracted_dates_rev = [x.split('/') for x in found_regulations if re.search('[0-9]+/[0-9]+', x)]
        extracted_dates = [x[::-1] for x in extracted_dates_rev]
        clean_dates = ['/'.join([x.translate(_trans_punc).strip() for x in y]) for y in extracted_dates]

        return clean_dates

    def get_text_by_tuple_index(self, input_tuple_list):
        """
        Given the input html and a list of tuples.. returns the text where the tuples define the range

        :param input_tuple_list: list with tuples
        :return: list with text
        """
        output_content = [[x.text for x in self.input_class_true[i_tuple[0]:i_tuple[1]]] for i_tuple in
                          input_tuple_list]
        return output_content

    def get_text_by_keyword(self, input_keyword):
        """
        Returns pieces of text if it contains the input_keyword-word. Using regex with IGNORECASE

        :param input_keyword:
        :return:
        """
        import re
        return [x.text for x in self.input_class_true if re.findall(input_keyword, x.text, re.IGNORECASE)]

    def get_text_by_class(self, class_name):
        """
        Returns the text content of a class..

        :param class_name: string name that represents a class in the html
        :return: a list with the content of this class name
        """
        return sorted([(i, class_name, x.text) for i, x in enumerate(self.input_class_true) if x['class'][0] ==
                       class_name])

    def get_text_context_class(self, class_name, n=1):
        """
        Returns the text given the class name. Here n defines the context in which we return text.. so n up or below
        the found class names (can be multiple occurrences of course
        :param class_name: a string that is present
        :param n:
        :return:
        """
        length_input = len(self.input_class_true) - 1
        output_list = []
        # text_content = [x.text for x in enumerate(input_list) if x['class'][0] == class_name]
        for j, x in enumerate(self.input_class_true):
            if x['class'][0] == class_name:
                temp_index_set = sorted(set([min(length_input, max(0, x)) for x in range(j-n, j+n+1)]))
                # print(temp_index_set)
                temp = [self.input_class_true[i].text for i in temp_index_set]
                output_list.append(temp)
        return output_list
