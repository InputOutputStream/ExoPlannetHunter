# csv_to_semicolon.py
import csv
import sys

if len(sys.argv) < 3:
    print("Usage: python3 csv_to_semicolon.py input.csv output.csv")
    sys.exit(2)

infile, outfile = sys.argv[1], sys.argv[2]

with open(infile, newline='', encoding='utf-8') as fin, \
     open(outfile, 'w', newline='', encoding='utf-8') as fout:
    reader = csv.reader(fin, delimiter=',', quotechar='"', doublequote=True)
    writer = csv.writer(fout, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True)
    for row in reader:
        writer.writerow(row)

print(f"Done. Wrote: {outfile}")
