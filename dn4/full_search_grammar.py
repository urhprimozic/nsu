from ProGED.generators.grammar import GeneratorGrammar

class full_search_grammar(GeneratorGrammar):
    '''
    Extends generator grammars from proged by adding a function for generating all the trees of depth below the given number. 
    Usefull when abusing ProGED to serve just as an equatiion generator. 
    '''
    def generate_all(self, max_depth):
        '''
        Returns a list of words for which the parse tree depth is at most `max_depth`. 
        '''
        pass
