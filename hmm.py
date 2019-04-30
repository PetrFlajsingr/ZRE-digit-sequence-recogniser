from math import log


class RecognisedWord:
    def __init__(self, word: str, time_start):
        self.word = word
        self.time_start = time_start
        self.time_end = time_start

    def end(self, time_end):
        self.time_end = time_end


class Token:
    def __init__(self, likelihood, history=None):
        if history is None:
            history = []
        self.likelihood = likelihood
        self.history = history

    def clone(self):
        return Token(self.likelihood, self.history.copy())


class State:
    def __init__(self, phoneme, is_word_start, is_word_end, transitions=None, word=''):
        if transitions is None:
            transitions = []
        self.phoneme = phoneme
        self.is_word_start = is_word_start
        self.is_word_end = is_word_end
        self.transitions = transitions
        self.word = word
        #self.tokens = [Token(float("-inf"))]
        self.tokens = []
        self.best_token = Token(float('-inf'))

    def purge_tokens(self):
        max = float('-inf')
        best = None
        for token in self.tokens:
            if token.likelihood >= max:
                max = token.likelihood
                best = token
        self.best_token = best
        self.tokens.clear()


class Transition:
    def __init__(self, probability, to):
        self.probability = probability
        self.to = to


class HMM:
    words = {}

    def __init__(self, phonemes, dictionary: (str, list)):
        self.phonemes = phonemes
        self.dictionary = dictionary
        self.states = []
        self.first = None
        self.last = None
        self.cnt = 0

    def build_network(self):
        log_0_5 = log(0.5)
        pause = State('pau', True, True)
        pause.transitions.append(Transition(log_0_5, pause))
        self.first = pause
        self.last = State('pau', False, False)
        self.states.append(pause)
        for word in self.dictionary:
            added = False
            last_state = None
            length = len(word[1]) - 1
            cnt = 0
            for phoneme in word[1]:
                new_state = State(phoneme, cnt == 0, cnt == length, word=word[0])
                cnt += 1
                self.states.append(new_state)
                new_state.transitions.append(Transition(log_0_5, new_state))
                if not added:
                    added = True
                    pause.transitions.append(Transition(log_0_5, new_state))
                else:
                    last_state.transitions.append(Transition(log_0_5, new_state))
                last_state = new_state
            last_state.transitions.append(Transition(log_0_5, self.last))
        self.states.append(self.last)
        self.states[0].best_token = Token(0.0)
        #self.states.reverse()

    def print(self):
        for t in self.states[0].transitions:
            self.__print(t.to)

    def __print(self, state):
        print(state.phoneme)
        if len(state.transitions) > 1:
            print("-", state.transitions[0].to.phoneme)
            self.__print(state.transitions[1].to)

    def step(self, likelihoods):
        for state in self.states:
            likelihood = likelihoods[self.phonemes.index(state.phoneme)]
            for t in state.transitions:
                new_token = state.best_token.clone()
                new_token.likelihood += likelihood + t.probability
                if state.is_word_start and t.to != state:
                    new_token.history.append(RecognisedWord(state.word, self.cnt))
                if state.is_word_end and t.to != state and len(new_token.history) > 0:
                    new_token.history[-1].time_end = self.cnt
                t.to.tokens.append(new_token)

        self.first.tokens.append(self.last.best_token.clone())
        for state in self.states:
            state.purge_tokens()

        self.cnt += 1

    def print_result(self):
        for word in self.states[-1].best_token.history:
            print(word.word, word.time_start, word.time_end)
