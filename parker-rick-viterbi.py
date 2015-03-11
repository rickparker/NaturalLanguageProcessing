import cPickle as pickle
import re
import sys, getopt

class HMM():
  # Utility tags to mark start and finish states
  start_line_marker = '_start_'
  finish_line_marker = '_finish_'

  # State set Q
  tag_states = []

  # Transition Probabilities A
  total_words = 0
  tag_unigrams = []
  tag_bigrams = []
  tag_trigrams = []
  lambda1 = 0.0
  lambda2 = 0.0
  lambda3 = 0.0

  # Observation Probabilities B, words for each tag
  tags_to_wordlist = {}

  # Track singleton words, to count part of speech distribution
  singleton_words = set()
  singleton_words_to_POS = {}
  multiple_words =set()
  unknown_words = []

  def __init__(self):
    self.load_tags()
    self.tag_unigrams = [0 for x in range(0, len(self.tag_states))]
    self.tag_bigrams = [[0 for x in range(0, len(self.tag_states))] \
                           for x in range(0, len(self.tag_states))]
    self.tag_trigrams = [[[0 for x in range(0, len(self.tag_states))] \
                             for x in range(0, len(self.tag_states))] \
                             for x in range(0, len(self.tag_states))]

    self.tags_to_wordlist = {}
    for x in self.tag_states:
      self.tags_to_wordlist[x] = {}
    self.singleton_words = set()
    self.singleton_words_to_POS = {}
    self.multiple_words = set()
    self.unknown_words = [0 for x in range (0, len(self.tag_states))]

  def load_tags(self):
    if len(self.tag_states) == 0:
      tags = [line.strip() for line in open('tagset.txt', 'r')]
      for x in tags:
        self.tag_states.append(x)
      self.tag_states.append(self.start_line_marker)
      self.tag_states.append(self.finish_line_marker)
      self.tag_states.sort()

  def learn_words(self, tagged_words):
    self.load_tags()
    tag_2 = self.tag_states.index(self.start_line_marker)
    tag_1 = self.tag_states.index(self.start_line_marker)
    tag_0 = self.tag_states.index(self.start_line_marker)
    for x in range(0, len(tagged_words) + 1):
      # Just processed last word of sentence?
      if x == len(tagged_words) and \
           tag_0 != self.tag_states.index(self.start_line_marker):
        tag_2 = tag_1
        tag_1 = tag_0
        tag_0 = self.tag_states.index(self.finish_line_marker)
        # No unigram storage
        self.tag_bigrams[tag_1][tag_0] += 1
        self.tag_trigrams[tag_2][tag_1][tag_0] += 1
        continue
      self.total_words += 1
      word_tag = tagged_words[x].split('\t')
      word = word_tag[0]
      tag = word_tag[1]

      tag_2 = tag_1
      tag_1 = tag_0
      tag_0 = self.tag_states.index(tag)
      self.tag_unigrams[tag_0] += 1
      self.tag_bigrams[tag_1][tag_0] += 1
      self.tag_trigrams[tag_2][tag_1][tag_0] += 1
      tag_index = self.tag_states.index(tag)
      if self.tags_to_wordlist[tag].has_key(word):
        self.tags_to_wordlist[tag][word] += 1
      else:
        self.tags_to_wordlist[tag][word] = 1

      if word in self.singleton_words:
        self.multiple_words.add(word)
        self.singleton_words.discard(word)
        del self.singleton_words_to_POS[word]
      elif word not in self.multiple_words:
        self.singleton_words.add(word)
        self.singleton_words_to_POS[word] = tag

  def prepare_unknowns(self):
    self.load_tags()
    print 'Analyze %d unknown words.' % (len(self.singleton_words))
    self.unknown_words = [0 for x in range (0, len(self.tag_states))]

    for x in self.singleton_words_to_POS:
      self.unknown_words[self.tag_states.index(self.singleton_words_to_POS[x])] += 1
    for x in self.tag_states:
      print 'Number of unknown %s: %d' % (x, self.unknown_words[self.tag_states.index(x)])

  def deleted_interpolation(self):
    self.load_tags()

    self.lambda1 = 0
    self.lambda2 = 0
    self.lambda3 = 0
    total = 0

    for x in range (0, len(self.tag_states)):
      for y in range (0, len(self.tag_states)):
        for z in range (0, len(self.tag_states)):
          uniprob = 0.0
          biprob = 0.0
          triprob = 0.0
          if self.tag_trigrams[x][y][z] > 0:
            uniprob = (self.tag_unigrams[z] - 1.0) / (self.total_words - 1.0)
            if uniprob > 0.0:
              biprob = (self.tag_bigrams[y][z] - 1.0) / (self.tag_unigrams[z] - 1.0)
            if biprob > 0.0:
              triprob = (self.tag_trigrams[x][y][z] - 1.0) / (self.tag_bigrams[y][z] - 1.0)
            if triprob > biprob and triprob > uniprob:
              self.lambda3 += self.tag_trigrams[x][y][z]
              total += self.tag_trigrams[x][y][z]
            elif biprob > triprob and biprob > uniprob:
              self.lambda2 += self.tag_bigrams[y][z]
              total += self.tag_bigrams[y][z]
            else:
              self.lambda1 += self.tag_unigrams[z]
              total += self.tag_unigrams[z]
    print 'Before normalization: lambda1 %d lambda2 %d lambda3 %d' % (self.lambda1, self.lambda2, self.lambda3)
    self.lambda1 = 1.0 * self.lambda1 / (1.0 * total)
    self.lambda2 = 1.0 * self.lambda2 / (1.0 * total)
    self.lambda3 = 1.0 * self.lambda3 / (1.0 * total)
    print 'After normalization: lambda1 %f lambda2 %f lambda3 %f' % (self.lambda1, self.lambda2, self.lambda3)

  def fetch_observation_likelihood(self, state_index, word):
    count = 0
    if self.tag_states[state_index] in self.tags_to_wordlist and \
         word in self.tags_to_wordlist[self.tag_states[state_index]]:
      count = self.tags_to_wordlist[self.tag_states[state_index]][word]
    else:
      # Handle unknown word
      count = self.unknown_words[state_index]

    if self.tag_unigrams[state_index] > 0:
      return (count * 1.0) / (self.tag_unigrams[state_index] * 1.0)
    else:
      return 0.0

  def fetch_transition_probability(self, state_0, state_1, state_2):
    count = 0
    single_prob = 0.0
    dual_prob = 0.0
    tri_prob = 0.0
    if self.total_words > 0:
      single_prob = self.tag_unigrams[state_2] * 1.0 / (self.total_words * 1.0)
    if self.tag_unigrams[state_2] > 0:
      dual_prob = self.tag_bigrams[state_1][state_2] * 1.0 / \
                 (self.tag_unigrams[state_2] * 1.0)
    if self.tag_bigrams[state_1][state_2] > 0:
      tri_prob = self.tag_trigrams[state_0][state_1][state_2] * 1.0 / \
                (self.tag_bigrams[state_1][state_2] * 1.0)
    return self.lambda1 * single_prob + \
           self.lambda2 * dual_prob + \
           self.lambda3 * tri_prob

  # observations: list of t observations to decode
  # returns sequence of states corresponding to decisions
  def viterbi_decode(self, observations):
    self.load_tags()
    # Create a path probability matrix viterbi(N+2, T)
    viterbi_matrix = [[0.0 for x in range (0, len(observations) + 2)] \
                         for x in range (0, len(self.tag_states))]
    back_references = [[0 for x in range (0, len(observations) + 2)] \
                        for x in range (0, len(self.tag_states))]
    # Initialization Step
    for s in range(0, len(self.tag_states)):
      viterbi_matrix[s][0] = \
        self.fetch_transition_probability( \
          self.tag_states.index(self.start_line_marker), \
          self.tag_states.index(self.start_line_marker), s) * \
        self.fetch_observation_likelihood(s, observations[0])
      back_references[s][0] = self.tag_states.index(self.start_line_marker)

    # Recursion Step
    for t in range(1, len(observations)):
      for s in range(0, len(self.tag_states)):
        X = [0.0 for x in range (0, len(self.tag_states))]
        max_index = -1
        max_value = -1.0

        for s_prime in range(0, len(self.tag_states)):
          X[s_prime] = viterbi_matrix[s_prime][t-1] * \
            self.fetch_transition_probability(back_references[s_prime][t-1], \
                                              s_prime, s)
          if X[s_prime] > max_value:
            max_value = X[s_prime]
            max_index = s_prime

        viterbi_matrix[s][t] = max_value * \
          self.fetch_observation_likelihood(s, observations[t])
        back_references[s][t] = max_index

    # Termination Step
    max_index = -1
    max_value = -1.0
    for s in range(0, len(self.tag_states)):
      value = viterbi_matrix[s][len(observations)] * \
              self.tag_bigrams[s][self.tag_states.index(self.finish_line_marker)]
      if value > max_value:
        max_value = value
        max_index = s
    viterbi_matrix[self.tag_states.index(self.finish_line_marker)]\
                  [len(observations)] = max_value
    back_references[self.tag_states.index(self.finish_line_marker)]\
                   [len(observations)] = max_index

    output = ['' for x in range (0, len(observations) + 1)]

    # Build output sequence by tracing backpointers of the highest
    # probabilities at each step
    current_tag = self.tag_states.index(self.finish_line_marker)
    for t in range (len(observations)-1, -1, -1):
      next_tag = back_references[current_tag][t+1]
      output[t] = self.tag_states[next_tag]
      current_tag = next_tag
    return output

def main(argv):
  if len(sys.argv) == 1:
    print 'Usage: python %s [-b <trainfile>] | [-m modelfile>] | [-v <trainfile>] | [-t <testfile>] | [-o <outputfile>' % argv[0]
    print 'Expected usage: Initially, run with -b and a <trainfile> to construct a model. Combine -b with -m and a <modelfile> to store the resulting model. Use -m and -t with a testing file to evaluate the model against the testing data, combined with -o to capture the results of the testing into an <outputfile>. Use -v and a trainfile to perform 10-fold cross-validation using the provided training file (all models will be discarded).'
    sys.exit(2)
  trainfile = ''
  modelfile = ''
  testfile = ''
  outputfile = ''
  do_validation = False

  try:
    opts, args = getopt.getopt(argv[1:], "hb:m:v:t:o:")
  except getopt.GetoptError:
    print 'Usage: python %s [-b <trainfile>] | [-m modelfile>] | [-v <trainfile>] | [-t <testfile>] | [-o <outputfile>' % argv[0]
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'Usage: python %s [-b <trainfile>] | [-m modelfile>] | [-v <trainfile>] | [-t <testfile>] | [-o <outputfile>' % argv[0]
      print 'Expected usage: Initially, run with -b and a <trainfile> to construct a model. Combine -b with -m and a <modelfile> to store the resulting model. Use -m and -t with a testing file to evaluate the model against the testing data, combined with -o to capture the results of the testing into an <outputfile>. Use -v and a trainfile to perform 10-fold cross-validation using the provided training file (all models will be discarded).'
      sys.exit()
    elif opt == '-b':
      trainfile = arg
    elif opt == '-m':
      modelfile = arg
    elif opt == '-v':
      trainfile = arg
      do_validation = True
    elif opt == '-t':
      testfile = arg
    elif opt == '-o':
      outputfile = arg
    else:
      print 'Unrecognized option \'%s\' : \'%s\'; disregarding.' % (opt, arg)

  if do_validation and trainfile != '':
    # if -v set, it overrides everything else.
    print 'Performing 10-fold cross-validation using %s.' % (trainfile)
  elif testfile != '' and modelfile != '':
    # if -t is set, will attempt to run testing mode
    print 'Performing test run of model %s against test set %s, results to %s.' % (modelfile, testfile, outputfile)
    model = pickle.load(open(modelfile, 'rb'))
    file_object = open(testfile, 'r')
    test_data = file_object.read()

    sentence_pattern = re.compile(r'\n\n')
    sentence_list = sentence_pattern.split(test_data)
    num_sentences = len(sentence_list)
    output_data = ''

    for x in range(0, num_sentences):
      if len(sentence_list[x]) > 0:
        word_list = sentence_list[x].split('\n')
        if len(word_list) > 0:
          tag_list = model.viterbi_decode(word_list)
        for y in range (0, len(word_list)):
          output_data += word_list[y] + '\t' + tag_list[y] + '\n'
      if x < num_sentences - 1:
        output_data += '\n'

    file_object = open(outputfile, 'w')
    file_object.write(output_data)
    file_object.close()

    print 'Processed %d sentences, results stored in %s.' % (num_sentences, outputfile)

  elif trainfile != '':
    # try to fall back to training mode
    print 'Performing model training using training set %s.' % (trainfile)
    model = HMM()
    file_object = open(trainfile, 'r')
    train_data = file_object.read()

    sentence_pattern = re.compile(r'\n\n')
    sentence_list = sentence_pattern.split(train_data)
    num_sentences = len(sentence_list)

    for x in range (0, num_sentences):
      if len(sentence_list[x]) > 0:
        word_list = sentence_list[x].split('\n')
        if len(word_list) > 0:
          model.learn_words(word_list)
    print 'Model has been fed all training sentences.'

    model.prepare_unknowns()
    model.deleted_interpolation()

    print 'Model smoothed and adjusted to handle unknowns.'

    if modelfile != '':
      print 'Will store model as %s.' % (modelfile)
      pickle.dump(model, open(modelfile, 'wb'))
    else:
      print 'Will discard model (specify modelfile using -m parameter).'
  else:
    print 'Did not recognize a training, testing, nor validation mode.'
    sys.exit()

if __name__ == "__main__":
  main(sys.argv)
