import itertools as it

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def get_variables_in_func_str(string_descr):
    
    varset = set()
    arithmetics = ["**", "*", "/", "+", "-"]
    
    def resolve_symbols_rec(string, sym_index, varset):
        if sym_index == len(arithmetics):
            varset.add(string)
            
        else:
            sym = arithmetics[sym_index]
            if sym in string:
                components = string.split(sym)
                for c in components:
                    resolve_symbols_rec(c, sym_index + 1, varset)
            else:
                resolve_symbols_rec(string, sym_index + 1, varset)

    
    # clean original string by removing parantheses, which are not relevant for this question
    string_descr = string_descr.replace("(", "").replace(")", "")
    
    # get symbols
    resolve_symbols_rec(string_descr, 0, varset)
    
    # clean symbols and eliminate numbers
    varset = {v.strip() for v in varset if v.strip() != "" and not is_number(v.strip())}
    
    # return
    return varset

def get_func_from_str(string_descr, order):
    
    # create list of parameters
    func_vars = ["x" + str(i) for i in range(1, order + 1)] + ["t"]
    for var in get_variables_in_func_str(string_descr):
        var = var.strip()
        if not var[0] == "x" and var != "t":
            func_vars.append(var)
    
    # string function description
    func_descr = "lambda " + ", ".join(func_vars) + ": " + string_descr
    return eval(func_descr)

class NaiveOptimizer:
    
    def __init__(self, num_vars, max_factors, max_summands):
        self.num_vars = num_vars
        self.max_factors = max_factors
        self.max_summands = max_summands

    def get_atomic_terms(self):
        var_names = ["t"] + [f"x{i}" for i in range(1, self.num_vars + 1)]

        def convert_list_to_term(l):
            syms = sorted(list(set(l)))
            powers = [len([e for e in l if e == s]) for s in syms]
            terms = []
            for s, p in zip(syms, powers):
                if p >= 2:
                    terms.append(f"({s}**{p})")
                elif p>= 1:
                    terms.append(f"{s}")
            return " * ".join(terms)

        def get_atomic_terms_rec(num_vars, max_factors, term):
            for var in var_names:
                if not term or var >= term[-1]:
                    new_term = term.copy()
                    new_term.append(var)
                    yield convert_list_to_term(new_term)

                    if max_factors > 1:
                        gen = get_atomic_terms_rec(num_vars, max_factors - 1, new_term)
                        for cand in gen:
                            yield convert_list_to_term(cand) if type(cand) == list else cand

        return get_atomic_terms_rec(self.num_vars, self.max_factors, [])


    def get_all_models_for_one_variable(self, constant_prefix):

        # first return single terms
        for num_summands in range(1, self.max_summands + 1):
            gens = [self.get_atomic_terms() for i in range(num_summands)]
            combo_gen = it.product(*gens)
            for combo in combo_gen:

                # ignore this if there are double terms
                if len(list(set(combo))) == len(combo):
                    yield " + ".join([f"({constant_prefix}{i} * {t})" for i, t in enumerate(combo)])

    def get_all_models(self):

        # constants for the different equations
        constant_space = ["a", "b", "c", "d", "e", "f"]

        # create generator for each variable
        gens = [self.get_all_models_for_one_variable(constant_space[i]) for i in range(self.num_vars)]
        return it.product(*gens)