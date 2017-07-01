import nltk


def store_sentences(filename, dump_filename):
    with open(filename, mode='r', encoding='utf-8') as file, open(dump_filename, mode='w', encoding='utf-8') as dump_file:
        for line in file.readlines():
            for item in nltk.sent_tokenize(line):
                idx = str(item).rfind(':')
                if idx != -1:
                    item = item[idx + 1:]
                # print(item)
                dump_file.write(item.strip() + '\n')


if __name__ == '__main__':
    store_sentences('data_new/articles.txt', 'data_new/sentences.txt')
