from math import log


class RecognisedWord:
    """
    Word record saved in tokens history. Contains information about starting and ending frame.
    """
    def __init__(self, word: str, time_start):
        self.word = word
        self.time_start = time_start
        self.time_end = time_start

    def end(self, time_end):
        self.time_end = time_end


class Token:
    """
    Representation of token for token passing. Contains history of words and its likelihood.
    """
    def __init__(self, likelihood, history=None):
        if history is None:
            history = []
        self.likelihood = likelihood
        self.history = history

    def clone(self):
        """
        Clone token and its history. Used for token passing.
        """
        return Token(self.likelihood, self.history.copy())


class State:
    """
    State of HMM. Contains information about accepting phoneme likelihood index.

    Contains tokens and provides method for picking the most likely one.
    """
    def __init__(self, phoneme, is_word_start, is_word_end, triphone_index, transitions=None, word=''):
        if transitions is None:
            transitions = []
        self.phoneme = phoneme
        self.triphone_index = triphone_index
        self.is_word_start = is_word_start
        self.is_word_end = is_word_end
        self.transitions = transitions
        self.word = word
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
    """
    Representation o transition between hmm's states.
    """
    def __init__(self, probability, to):
        self.probability = probability
        self.to = to


class HMM:
    words = {}

    def __init__(self, phonemes, dictionary: (str, list)):
        self.phonemes = phonemes
        self.dictionary = dictionary
        self.states = []
        self.word_start_states = []
        self.first = None
        self.last = None
        self.cnt = 0

    def build_network(self):
        log_0_5 = log(0.5)

        pause_start = State('pau', True, False, self.__get_triphone_index('pau', 0), word='pause')
        pause_start.transitions.append(Transition(log_0_5, pause_start))
        self.word_start_states.append(pause_start)
        pause_start_2 = State('pau', False, False, self.__get_triphone_index('pau', 1), word='pause')
        pause_start.transitions.append(Transition(log_0_5, pause_start_2))
        pause_start_2.transitions.append(Transition(log_0_5, pause_start_2))
        pause_start_3 = State('pau', False, True, self.__get_triphone_index('pau', 1), word='pause')
        pause_start_2.transitions.append(Transition(log_0_5, pause_start_3))
        pause_start_3.transitions.append(Transition(log_0_5, pause_start_3))

        self.first = pause_start
        self.states.append(pause_start)
        self.states.append(pause_start_2)
        self.states.append(pause_start_3)

        pause_end = State('pau', True, False, self.__get_triphone_index('pau', 0), word='pause')
        pause_end.transitions.append(Transition(log_0_5, pause_end))
        pause_2 = State('pau', False, False, self.__get_triphone_index('pau', 1), word='pause')
        pause_end.transitions.append(Transition(log_0_5, pause_2))
        pause_2.transitions.append(Transition(log_0_5, pause_2))
        pause_3 = State('pau', False, True, self.__get_triphone_index('pau', 2), word='pause')
        pause_2.transitions.append(Transition(log_0_5, pause_3))
        pause_3.transitions.append(Transition(log_0_5, pause_3))

        self.last = pause_end

        for word in self.dictionary:
            added = False
            last_state = None
            length = len(word[1]) - 1
            cnt = 0
            for phoneme in word[1]:
                new_state_1 = State(phoneme, cnt == 0, False, self.__get_triphone_index(phoneme, 0),
                                    word=word[0])
                self.word_start_states.append(new_state_1)
                new_state_1.transitions.append(Transition(log_0_5, new_state_1))
                self.states.append(new_state_1)
                new_state_2 = State(phoneme, False, False, self.__get_triphone_index(phoneme, 1),
                                    word=word[0])
                new_state_1.transitions.append(Transition(log_0_5, new_state_2))
                new_state_2.transitions.append(Transition(log_0_5, new_state_2))
                self.states.append(new_state_2)
                new_state_3 = State(phoneme, False, cnt == length, self.__get_triphone_index(phoneme, 2),
                                    word=word[0])
                new_state_2.transitions.append(Transition(log_0_5, new_state_3))
                new_state_3.transitions.append(Transition(log_0_5, new_state_3))
                self.states.append(new_state_3)

                cnt += 1
                if not added:
                    added = True
                    pause_start_3.transitions.append(Transition(log_0_5, new_state_1))
                else:
                    last_state.transitions.append(Transition(log_0_5, new_state_1))
                last_state = new_state_3
            last_state.transitions.append(Transition(log_0_5, self.last))

        self.states.append(pause_end)
        self.states.append(pause_2)
        self.states.append(pause_3)
        self.first.best_token = Token(0.0)

    def __get_triphone_index(self, phoneme, order_num):
        return self.phonemes.index(phoneme) * 3 + order_num

    def print(self):
        for t in self.states[0].transitions:
            self.__print(t.to)

    def __print(self, state):
        print(state.phoneme)
        for t in state.transitions:
            print("-", t.to.phoneme)
            if t.to != state:
                self.__print(t.to)

    def step(self, likelihoods):
        for state in self.states:
            likelihood = likelihoods[state.triphone_index]
            for t in state.transitions:
                new_token = state.best_token.clone()
                new_token.likelihood += likelihood + t.probability
                if state.is_word_start and t.to != state:
                    new_token.history.append(RecognisedWord(state.word, self.cnt))
                if state.is_word_end and t.to != state and len(new_token.history) > 0:
                    new_token.history[-1].time_end = self.cnt
                t.to.tokens.append(new_token)

        for state in self.states:
            state.purge_tokens()
        self.first.tokens.append(self.last.best_token.clone())
        self.first.purge_tokens()

        self.cnt += 1

    def print_result(self, print_frames=False):
        for word in self.states[-1].best_token.history:
            if word.word == 'pause':
                continue
            print(word.word)
            if print_frames:
                print('\t' + str(word.time_start) + '\t' + str(word.time_end))
