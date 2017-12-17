import preprocess.tweettovect as tv

print ("Starting")
import csv
with open('labeled_data_txt_class.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        print (row)
print("Convert")
C = tv.CSVTweetsToMatrix("labeled_data_txt_class.csv", "labeled_tweets.csv", 280)
print ("Converting")
C.convert()
print("Done!")


