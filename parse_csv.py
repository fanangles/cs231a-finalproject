import csv

all_loc = []
with open('coords.csv') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		all_loc.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])

tids = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
classes = [4]
for tid in tids:
	for c in classes:
		file = open(str(tid) + '.csv', 'wb')
		writer = csv.writer(file)
		for loc in all_loc:
			if tid == loc[0]:
				if tid in [41, 42, 44, 45, 48, 49, 50]:
					writer.writerow((int(loc[2]/ 4.4), int(loc[3]/4.4)))
				else:
					writer.writerow((int(loc[2]/ 4.95), int(loc[3]/4.95)))
		file.close()