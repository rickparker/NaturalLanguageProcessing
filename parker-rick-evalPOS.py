import sys

if len(sys.argv) < 3 or len(sys.argv) > 4:
  print 'Usage: python %s goldPOS_filename evalPOS_filename [result_filename]' \
    % (sys.argv[0])
  sys.exit(1)

tags = [line.strip() for line in open('tagset.txt', 'r')]
gold = [line.strip() for line in open(sys.argv[1], 'r')]
output = [line.strip() for line in open(sys.argv[2], 'r')]

if len(gold) != len(output):
  print 'number of lines between gold and output do not match!'
  sys.exit(1)

blank_line_marker = '_bl_'

taglist = []
for x in tags:
  taglist.append(x)
taglist.append(blank_line_marker)
taglist.sort()

conf_matrix = [[0 for x in range(0, len(taglist))] \
                for x in range(0, len(taglist))]

for i in range (0, len(gold)):
  gold_index = 0
  output_index = 0
  if not gold[i]:
    gold_index = taglist.index(blank_line_marker)
  else:
    gold_index = taglist.index(gold[i].split('\t')[1])
  if not output[i]:
    output_index = taglist.index(blank_line_marker)
  else:
    output_index = taglist.index(output[i].split('\t')[1])

  conf_matrix[gold_index][output_index] += 1

# Dump the confusion matrix to out
report = 'Evaluating %s as GOLD against %s as OUTPUT.\nConfusion Matrix: (GOLD is rows, OUTPUT is columns)\n'

# Output labels
dumpstring = '\t'
for i in range (0, len(taglist)):
  dumpstring += taglist[i] + '\t'
dumpstring += 'Row Sum'
report += dumpstring + '\n'

# For each row (GOLD)
for i in range (0, len(taglist)):
  total = 0 # running sum
  dumpstring = taglist[i] + '\t' # Label
  # For each column
  for j in range (0, len(taglist)):
    total += conf_matrix[i][j]
    dumpstring += str(conf_matrix[i][j]) + '\t'
  dumpstring += str(total)
  report += dumpstring + '\n'

#Sum columns
dumpstring = 'Col Sum\t'
tablesum = 0
for j in range (0, len(taglist)):
  total = 0 # running sum
  for i in range (0, len(taglist)):
    total += conf_matrix[i][j]
  tablesum += total
  dumpstring += str(total) + '\t'
dumpstring += str(tablesum)
report += dumpstring + '\n'

if len(sys.argv) == 4:
  file_object = open(sys.argv[3], 'w')
  file_object.write(report)
  file_object.close()
  print 'Results stored to %s.' % (sys.argv[3])
else:
  print 'No destination set for results, dumping to stdout.'
  print report
