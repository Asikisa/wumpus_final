"""

    KB is an abstract class holds a knowledge base of logical expressions
    Expr is a logical expression, imported from utils.py
    substitution is an implemented as a dictionary of var:value pairs, {x:1, y:x}

"""

import itertools

# from agents import Agent, Glitter, Bump, Stench, Breeze, Scream

from utils import remove_all, first, Expr, expr, subexpressions, extend


class KB:
    """A knowledge base to which you can tell and ask sentences.
    To create a KB, first subclass this class and implement
    tell, ask_generator, and retract.
    For a Propositional Logic KB, ask(P & Q) returns True or False, but for an

    So ask_generator generates these one at a time, and ask either returns the
    first one or returns False."""

    def __init__(self, sentence=None):
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        # add the sentence to the KB
        raise NotImplementedError

    def ask(self, query):
        # return a substitution that makes the query true, or, failing that, return False
        return first(self.ask_generator(query), default=False)

    def ask_generator(self, query):
        # yield all the substitutions that make query true
        raise NotImplementedError

    def retract(self, sentence):
        # remove sentence from the KB
        raise NotImplementedError


def KBAgentProgram(kb):
    # a generic logical knowledge-based agent program.

    steps = itertools.count()

    def program(percept):
        t = next(steps)
        kb.tell(make_percept_sentence(percept, t))
        action = kb.ask(make_action_query(t))
        kb.tell(make_action_sentence(action, t))
        return action

    def make_percept_sentence(percept, t):
        return Expr('Percept')(percept, t)

    def make_action_query(t):
        return expr('ShouldDo(action, {})'.format(t))

    def make_action_sentence(action, t):
        return Expr('Did')(action[expr('action')], t)

    return program


def is_symbol(s):
    """A string s is a symbol if it starts with an alphabetic char.
    """
    return isinstance(s, str) and s[:1].isalpha()


def is_var_symbol(s):
    # logic variable symbol is an initial-lowercase string.

    return is_symbol(s) and s[0].islower()


def is_prop_symbol(s):
    # a proposition logic symbol is an initial-uppercase string.
    return is_symbol(s) and s[0].isupper()


def variables(s):
    # return a set of the variables in expression s.

    return {x for x in subexpressions(s) if is_variable(x)}


def prop_symbols(x):
    """Return the set of all propositional symbols in x."""
    if not isinstance(x, Expr):
        return set()
    elif is_prop_symbol(x.op):
        return {x}
    else:
        return {symbol for arg in x.args for symbol in prop_symbols(arg)}


def pl_true(exp, model={}):
    # return True if the propositional logic expression is true in the model,
    # and False if it is false. If the model does not specify the value for
    # every proposition, this may return None to indicate 'not obvious';
    # this may happen even when the expression is tautological.

    if exp in (True, False):
        return exp
    op, args = exp.op, exp.args
    if is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                result = None
        return result
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^':  # xor or 'not equivalent'
        return pt != qt
    else:
        raise ValueError('Illegal operator in logic expression' + str(exp))


# Convert to Conjunctive Normal Form (CNF)

def to_cnf(s):
    # Convert a propositional logical sentence to conjunctive normal form.
    # That is, to the form ((A | ~B | ...) & (B | C | ...) & ...)

    s = expr(s)
    if isinstance(s, str):
        s = expr(s)
    s = eliminate_implications(s)  # Steps 1, 2 from p. 253
    s = move_not_inwards(s)  # Step 3
    return distribute_and_over_or(s)  # Step 4


def eliminate_implications(s):
    # Change implications into equivalent form with only &, |, and ~ as logical operators
    s = expr(s)
    if not s.args or is_symbol(s.op):
        return s  # Atoms are unchanged.
    args = list(map(eliminate_implications, s.args))
    a, b = args[0], args[-1]
    if s.op == '==>':
        return b | ~a
    elif s.op == '<==':
        return a | ~b
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)


def move_not_inwards(s):
    # Rewrite sentence s by moving negation sign inward.

    s = expr(s)
    if s.op == '~':
        def NOT(b):
            return move_not_inwards(~b)

        a = s.args[0]
        if a.op == '~':
            return move_not_inwards(a.args[0])  # ~~A ==> A
        if a.op == '&':
            return associate('|', list(map(NOT, a.args)))
        if a.op == '|':
            return associate('&', list(map(NOT, a.args)))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *list(map(move_not_inwards, s.args)))


def distribute_and_over_or(s):
    # Given a sentence s consisting of conjunctions and disjunctions
    # of literals, return an equivalent sentence in CNF

    s = expr(s)
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        if len(s.args) == 0:
            return False
        if len(s.args) == 1:
            return distribute_and_over_or(s.args[0])
        conj = first(arg for arg in s.args if arg.op == '&')
        if not conj:
            return s
        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c | rest)
                               for c in conj.args])
    elif s.op == '&':
        return associate('&', list(map(distribute_and_over_or, s.args)))
    else:
        return s


def associate(op, args):
    # Given an associative op, return an expression with the same
    #  meaning as Expr(op, *args)

    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)


_op_identity = {'&': True, '|': False, '+': 0, '*': 1}


def dissociate(op, args):
    # Given an associative op, return a flattened list result such
    # that Expr(op, *result) means the same as Expr(op, *args)

    result = []

    def collect(subargs):
        for arg in subargs:
            if arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)

    collect(args)
    return result


def conjuncts(s):
    # Return a list of the conjuncts in the sentence s.

    return dissociate('&', [s])


def disjuncts(s):
    # Return a list of the disjuncts in the sentence s.

    return dissociate('|', [s])


def no_branching_heuristic(symbols, clauses):
    return first(symbols), True


# DPLL-Satisfiable

def dpll_satisfiable(s, branching_heuristic=no_branching_heuristic):
    # Check satisfiability of a propositional sentence.

    return dpll(conjuncts(to_cnf(s)), prop_symbols(s), {}, branching_heuristic)


def dpll(clauses, symbols, model, branching_heuristic=no_branching_heuristic):
    # See if the clauses are true in a partial model
    unknown_clauses = []  # clauses with an unknown truth value
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is None:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P, value = find_pure_symbol(symbols, unknown_clauses)
    if P:
        return dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic)
    P, value = find_unit_clause(clauses, model)
    if P:
        return dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic)
    P, value = branching_heuristic(symbols, unknown_clauses)
    return (dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic) or
            dpll(clauses, remove_all(P, symbols), extend(model, P, not value), branching_heuristic))


def find_pure_symbol(symbols, clauses):
    # Find a symbol and its value if it appears only as a positive literal
    # (or only as a negative) in clauses.

    for s in symbols:
        found_pos, found_neg = False, False
        for c in clauses:
            if not found_pos and s in disjuncts(c):
                found_pos = True
            if not found_neg and ~s in disjuncts(c):
                found_neg = True
        if found_pos != found_neg:
            return s, found_pos
    return None, None


def find_unit_clause(clauses, model):
    # Find a forced assignment if possible from a clause with only 1
    # variable not bound in the model
    for clause in clauses:
        P, value = unit_clause_assign(clause, model)
        if P:
            return P, value
    return None, None


#
def unit_clause_assign(clause, model):
    # Return a single variable/value pair that makes clause true in
    # the model, if possible.

    P, value = None, None
    for literal in disjuncts(clause):
        sym, positive = inspect_literal(literal)
        if sym in model:
            if model[sym] == positive:
                return None, None  # clause already True
        elif P:
            return None, None  # more than 1 unbound variable
        else:
            P, value = sym, positive
    return P, value


def inspect_literal(literal):
    # The symbol in this literal, and the value it should take to
    # make the literal true.

    if literal.op == '~':
        return literal.args[0], False
    else:
        return literal, True


class WumpusPosition:
    def __init__(self, x, y, orientation):
        self.X = x
        self.Y = y
        self.orientation = orientation

    def get_location(self):
        return self.X, self.Y

    def set_location(self, x, y):
        self.X = x
        self.Y = y

    def get_orientation(self):
        return self.orientation

    def set_orientation(self, orientation):
        self.orientation = orientation

    def __eq__(self, other):
        if other.get_location() == self.get_location() and other.get_orientation() == self.get_orientation():
            return True
        else:
            return False


def is_variable(x):
    # a variable is an Expr with no args and a lowercase symbol as the op
    return isinstance(x, Expr) and not x.args and x.op[0].islower()
