#coding:utf-8

import numpy as np

class DataLoader(object):
    """单个文件数据读取"""
    def __init__(self, datapath, wordlist):
        self.datapath = datapath
        self.wordlist = wordlist
        self.datas = self._load(datapath)


    def _load(self, datapath):
        raw_contents = []
        coded_contents = []
        aspects = []
        labels = []
        with open(datapath,'r') as f:
            sentences = f.readlines()
            for i in xrange(len(sentences) / 3):
                raw_content, aspect, label = sentences[i * 3].decode('utf-8').strip(), sentences[i * 3 + 1].decode('utf-8').strip(), sentences[ i * 3 + 2].strip()
                label = int(label)
                coded_content = self.sentenceCoding(raw_content)
                raw_contents.append(raw_content)
                coded_contents.append(coded_content)
                aspects.append(aspect)
                labels.append(label)
        return [raw_contents, coded_contents,aspects,labels]

    def sentenceCoding(self, sentence):
        words = sentence.split(' ')
        coded = []
        for word in words:
            if(self.wordlist.has_key(word)):
                id = self.wordlist[word]
                coded.append(id)
            else:
                coded.append(0)


        #return [self.wordlist[word] for word in words]
        return coded

    def getData(self):
        return self.datas



class DataManager(object):
    """文件组载入"""
    def __init__(self, dataset, grained=3):
        self.fileList = ['train', 'test', 'dev']
        self.wordlist = self.genwordlist(dataset)
        self.origin = {}
        for fname in self.fileList:
            contents = []
            aspects = []
            labels = []
            with open('%s/%s.cor' % (dataset, fname)) as f:
                sentences = f.readlines()
                for i in xrange(len(sentences) / 3):
                    content, aspect, label = sentences[i * 3].decode('utf-8').strip(), sentences[i * 3 + 1].decode('utf-8').strip(), sentences[
                        i * 3 + 2].strip()
                    label = int(label)
                    content = self.sentenceCoding(content)
                    contents.append(content)
                    aspects.append(aspect)
                    labels.append(label)

            self.origin[fname] = [contents,aspects,labels]

    def getTrainData(self):
        return self.origin['train']

    def getTestData(self):
        return self.origin['test']

    def getDevData(self):
        return self.origin['dev']

    def sentenceCoding(self, sentence):
        words = sentence.split(' ')
        return [self.wordlist[word] for word in words]



    def genwordlist(self,dataset):
        #self.fileList = ['train', 'test', 'dev']
        word_count = {}
        for fname in self.fileList:
            with open('%s/%s.cor' % (dataset, fname)) as f:
                sentences = f.readlines()
                for i in xrange(len(sentences) / 3):
                    content = sentences[i * 3].decode('utf-8').strip()
                    words = content.split(' ')
                    for word in words:
                        if word in word_count:
                            count = word_count[word]
                            word_count[word] = count+1
                        else:
                            word_count[word] = 1


        sorted_list = sorted(word_count.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        wordlist = {}
        for i in xrange(len(sorted_list)):
            wordlist[sorted_list[i][0]] = i
        return wordlist

    def getwordlist(self):
        return self.wordlist
