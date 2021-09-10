from collections import defaultdict
import argparse
class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(self, input, word_dict_min_num=2):
        self.files = ['train.0', 'dev.0', 'test.0', 'train.1', 'dev.1', 'test.1']
        self.files = [input+"/"+f for f in self.files]

        self.word2id = defaultdict(int)
        self.id2word = defaultdict(str)
        self.count = defaultdict(int)

        self.tokenizer = Tokenizer()
        self.bos, self.eos, self.pad, self.unk = "<s>", "</s>", "<pad>", "<unk>"
        self.add_symbol(self.bos)
        self.add_symbol(self.pad)
        self.add_symbol(self.eos)
        self.add_symbol(self.unk)
        
        self.read_data()
        self.finalize(word_dict_min_num)

    def read_data(self):
        for file_item in self.files:
            with open(file_item, 'r') as f:
                for item in f:
                    item = item.strip()
                    word_list = item.split(" ")
                    for word in word_list:
                        word = word.lower()
                        self.count[word] += 1

    def add_symbol(self, word):
        idx = len(self.word2id)
        self.word2id[word] = idx
        self.id2word[idx] = word


    def finalize(self, word_dict_min_num):
        for word in sorted(self.count, key = lambda x: self.count[x], reverse = True):
            if self.count[word] >= word_dict_min_num:
                self.add_symbol(word)


    def convert_ids_to_string(self, ids):
        sen_text = []
        max_i = len(self.word2id)
        for i in ids:
            if i == self.get_eos():
                break
            sen_text.append(self.id2word[i])
        return ' '.join(sen_text)

    def get_id_files(self, max_sequence_length, file_type):
        if file_type in ["train", "dev", "test"]:
            files = [file_item for file_item in self.files if file_type in file_item]
        else:
            files = [file_type]

        pairs = []
        for ind, file_item in enumerate(files):
            with open(file_item) as f:
                item = f.readline()
                while(item):
                    item = item.strip()
                    word_list = self.tokenizer.tokenize(item)
                    id_list = self.convert_string_to_ids(word_list[:max_sequence_length])
                    pairs.append([id_list, [ind]] if not self.label else [id_list, [self.label]])
                    item = f.readline()
        
        return pairs
                    
    def convert_string_to_ids(self, strings):
        id_list = []
        for word in strings:
            word = word.lower()            
            if word not in self.word2id:
                id = self.get_unk()
            else:
                id = self.word2id[word]
            id_list.append(id)
        return id_list

    def set_pretrained_vocab(self, pretrained_model):
        for word, ind in self.tokenizer.vocab.items():
            self.word2id[word] = ind
            self.id2word[ind] = word
            if word in ["[CLS]", "<s>"]:
                self.bos = word
            elif word in ["[SEP]", "</s>"]:
                self.eos = word
            elif word in ["[PAD]", "<pad>"]:
                self.pad = word
            elif word in ["[UNK]", "<unk>"]:
                self.unk = word


    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.word2id)

    def get_bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.word2id[self.bos]

    def get_pad(self):
        """Helper to get index of pad symbol"""
        return self.word2id[self.pad]

    def get_eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.word2id[self.eos]

    def get_unk(self):
        """Helper to get index of unk symbol"""
        return self.word2id[self.unk]

    def swap_id(self, id1, id2):
        word1 = self.id2word[id1]
        word2 = self.id2word[id2]
        self.id2word[id1] = word2
        self.id2word[id2] = word1
        self.word2id[word1] = id2
        self.word2id[word2] = id1


    def keep_same_special_symbols(self, dictionary):
        if self.bos != dictionary.bos or self.get_bos()!= dictionary.get_bos():
            ind = self.word2id[self.bos]
            self.word2id.pop(self.bos)
            self.bos = dictionary.bos
            self.word2id[self.bos] = ind
            self.id2word[ind] = self.bos
            if ind != dictionary.get_bos():
                self.swap_id(ind, dictionary.get_bos())

        if self.eos != dictionary.eos or self.get_eos()!= dictionary.get_eos():
            ind = self.word2id[self.eos]
            self.word2id.pop(self.eos)
            self.eos = dictionary.eos
            self.word2id[self.eos] = ind
            self.id2word[ind] = self.eos
            if ind != dictionary.get_eos():
                self.swap_id(ind, dictionary.get_eos())

        if self.pad != dictionary.pad or self.get_pad()!= dictionary.get_pad():
            ind = self.word2id[self.pad]
            self.word2id.pop(self.pad)
            self.pad = dictionary.pad
            self.word2id[self.pad] = ind
            self.id2word[ind] = self.pad
            if ind != dictionary.get_pad():
                self.swap_id(ind, dictionary.get_pad())

        if self.unk != dictionary.unk or self.get_unk()!= dictionary.get_unk():
            ind = self.word2id[self.unk]
            self.word2id.pop(self.unk)
            self.unk = dictionary.unk
            self.word2id[self.unk] = ind
            self.id2word[ind] = self.unk
            if ind != dictionary.get_unk():
                self.swap_id(ind, dictionary.get_unk())
        

class Tokenizer:
    def tokenize(self, strings):
        return strings.lower().split(' ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input folder')
    parser.add_argument('--output', type=str, required=True, help='output file')
    args = parser.parse_args()
    d = Dictionary(args.input, word_dict_min_num=2)
    with open(args.output, "w") as f:
        for i in range(len(d)):
            f.writelines([d.id2word[i], "\n"])