import re
import sys, getopt

def main(argv):
  inputfile = ''
  try:
    opts, args = getopt.getopt(argv, "hi:o:","ifile=")
  except getopt.GetoptError:
    print 'parker-rick-assgn1.py -i <inputfile>'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'parker-rick-assgn1.py -i <inputfile>'
      sys.exit()
    elif opt in ("-i", "--ifile"):
      inputfile = arg

  if inputfile == '':
    print 'No inputfile specified, aborting.'
    print 'parker-rick-assgn1.py -i <inputfile>'
    sys.exit(2)

  print 'Input file is ', inputfile

  file_object = open(inputfile, 'r')
  data = file_object.read()
  data += '\n\n'  # Ensure file ends with carriage returns

  paragraph_list = re.findall(r'(\w|\W)\W*\n\W*\n', data)

  num_paragraphs = len(paragraph_list)

  sentence_list = re.findall(r'(\w|\W)*?[\.|\?|\!](\s+|\"\s+|\)\s+)', data)

  num_sentences = len(sentence_list)

  word_list = re.findall(r'(\w|\W)+?( |\n|$)', data)

  num_words = len(word_list)

  print "Paragraph count: %d\nSentence count: %d\nWord count: %d" \
    % (num_paragraphs, num_sentences, num_words)

if __name__ == "__main__":
  main(sys.argv[1:])
