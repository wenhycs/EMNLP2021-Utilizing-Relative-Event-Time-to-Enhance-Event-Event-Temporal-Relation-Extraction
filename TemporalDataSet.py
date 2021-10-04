import torch
import xml.etree.ElementTree as ET

from torch.utils.data import TensorDataset 

from utils import LabelType

class temprel_ee:
    def __init__(self, xml_element):
        self.xml_element = xml_element
        self.label = xml_element.attrib['LABEL']
        self.sentdiff = int(xml_element.attrib['SENTDIFF'])
        self.docid = xml_element.attrib['DOCID']
        self.source = xml_element.attrib['SOURCE']
        self.target = xml_element.attrib['TARGET']
        self.data = xml_element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        self.event_ix = []
        self.text = ""
        self.event_offset = []

        is_start = True
        for i,d in enumerate(self.data):
            tmp = d.split('///')
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])

            if is_start:
                is_start = False
            else:
                self.text += " "
            
            if tmp[-1] == 'E1':
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))
            elif tmp[-1] == 'E2':
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))
            
            self.text += tmp[0]

        assert len(self.event_ix) == 2


class temprel_set:
    def __init__(self, xmlfname, datasetname="matres"):
        self.xmlfname = xmlfname
        self.datasetname = datasetname
        tree = ET.parse(xmlfname)
        root = tree.getroot()
        self.size = len(root)
        self.temprel_ee = []
        for e in root:
            self.temprel_ee.append(temprel_ee(e))
    
    def to_tensor(self, tokenizer):
        gathered_text = [ee.text for ee in self.temprel_ee]
        tokenized_output = tokenizer(gathered_text, padding=True, return_offsets_mapping=True)
        tokenized_event_ix = []
        for i in range(len(self.temprel_ee)):
            event_ix_pair = []
            for j, offset_pair in enumerate(tokenized_output['offset_mapping'][i]):
                if (offset_pair[0] == self.temprel_ee[i].event_offset[0] or\
                    offset_pair[0] == self.temprel_ee[i].event_offset[1]) and\
                   offset_pair[0] != offset_pair[1]:
                    event_ix_pair.append(j)
            if len(event_ix_pair) != 2:
                raise ValueError(f'Instance {i} doesn\'t found 2 event idx.')
            tokenized_event_ix.append(event_ix_pair)
        input_ids = torch.LongTensor(tokenized_output['input_ids'])
        attention_mask = torch.LongTensor(tokenized_output['attention_mask'])
        tokenized_event_ix = torch.LongTensor(tokenized_event_ix)
        labels = torch.LongTensor([LabelType.to_class_index(ee.label) for ee in self.temprel_ee])
        return TensorDataset(input_ids, attention_mask, tokenized_event_ix, labels)