import csv

class read_csv(object):
    def __init__(self):
        #train = file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
        self.data_path = ""

    # Reads a given CSV and stores the data in a list
    def read_csv(self, data_path, test_list=False):
        file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
        sent_list = []
        # print(file_reader.shape)
        # print(file_reader.columns.values)
        for row in file_reader:
            id = row[0]
            suggest = row[1]
            sent = row[2]
            if test_list == False:
                sent_list.append((id, suggest, sent))
            else:
                sent_list.append((id, suggest))
        return sent_list

    # This will create and write into a new CSV
    def write_csv(self, sent_list, label_list, out_path):
        filewriter = csv.writer(open(out_path, "w+"))
        count = 0
        for ((id, sent), label) in zip(sent_list, label_list):
            filewriter.writerow([id, sent, label])


read = read_csv()
