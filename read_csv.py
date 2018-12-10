import csv

class read_csv(object):
    def __init__(self):
        #train = file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
        self.data_path = "/home/paulo/PycharmProjects/suggestion-mining/training-full-v13-bkp.csv"

    # Reads a given CSV and stores the data in a list
    def read_csv(self):
        file_reader = csv.reader(open(self.data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
        sent_list = []
        # print(file_reader.shape)
        # print(file_reader.columns.values)
        for row in file_reader:
            id = row[0]
            suggest = row[1]
            sent = row[2]
            sent_list.append((id, suggest, sent))
        return sent_list

read = read_csv()