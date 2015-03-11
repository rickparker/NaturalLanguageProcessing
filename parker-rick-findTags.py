import sys

if len(sys.argv) != 2:
  print 'Usage: python %s tagsource_filename' % (sys.argv[0])
  sys.exit(1)

input_data = [line.strip() for line in open(sys.argv[1], 'r')]
tagset = set()

for i in range(0, len(input_data)):
  if input_data[i]:
    tagset.add(input_data[i].split('\t')[1]

taglist = []
for x in tagset:
  taglist.append(x)
taglist.sort()

output = ''
for x in taglist:
  output += x + '\n'

file_object = open('tagset.txt', 'w')
file_object.write(output)
file_object.close()

print 'Identified %d tags, stored to \'tagset.txt\'.' % len(taglist)
