import sys

if len(sys.argv) != 3:
  print 'Usage: python %s input_filename output_filename' % (sys.argv[0])
  sys.exit(1)

input_data = [line.strip() for line in open(sys.argv[1], 'r')]
output_data = ''

line_count = 0
for i in range(0, len(input_data)):
  if input_data[i]:
    line_count += 1
    output_data += input_data[i].split('\t')[0]
  if i != len(input_data) - 1:
    output_data += '\n'

file_object = open(sys.argv[2], 'w')
file_object.write(output_data)
file_object.close()

print 'Stripped tags from %d lines.' % line_count
