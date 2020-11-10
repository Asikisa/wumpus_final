import collections.abc


def remove_all(item, seq):
    # return a copy of seq (or string) with all occurrences of item removed
    if isinstance(seq, str):
        return seq.replace(item, '')
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    else:
        return [x for x in seq if x != item]


def first(iterable, default=None):
    # return the first element of an iterable or default
    return next(iter(iterable), default)


def extend(s, var, val):
    # copy dict s and extend it by setting var to val and return copy
    return {**s, var: val}


class Expr:
    # mathematical expression with an operator and 0 or more arguments

    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args

    def __neg__(self):
        return Expr('-', self)

    def __pos__(self):
        return Expr('+', self)

    def __invert__(self):
        return Expr('~', self)

    def __add__(self, rhs):
        return Expr('+', self, rhs)

    def __sub__(self, rhs):
        return Expr('-', self, rhs)

    def __mul__(self, rhs):
        return Expr('*', self, rhs)

    def __pow__(self, rhs):
        return Expr('**', self, rhs)

    def __mod__(self, rhs):
        return Expr('%', self, rhs)

    def __and__(self, rhs):
        return Expr('&', self, rhs)

    def __xor__(self, rhs):
        return Expr('^', self, rhs)

    def __rshift__(self, rhs):
        return Expr('>>', self, rhs)

    def __lshift__(self, rhs):
        return Expr('<<', self, rhs)

    def __truediv__(self, rhs):
        return Expr('/', self, rhs)

    def __floordiv__(self, rhs):
        return Expr('//', self, rhs)

    def __matmul__(self, rhs):
        return Expr('@', self, rhs)

    def __or__(self, rhs):
        """Allow both P | Q, and P |'==>'| Q."""
        if isinstance(rhs, Expression):
            return Expr('|', self, rhs)
        else:
            return PartialExpr(rhs, self)

    # # Reverse operator overloads
    # def __radd__(self, lhs):
    #     return Expr('+', lhs, self)
    #
    # def __rsub__(self, lhs):
    #     return Expr('-', lhs, self)
    #
    # def __rmul__(self, lhs):
    #     return Expr('*', lhs, self)
    #
    # def __rdiv__(self, lhs):
    #     return Expr('/', lhs, self)
    #
    # def __rpow__(self, lhs):
    #     return Expr('**', lhs, self)
    #
    # def __rmod__(self, lhs):
    #     return Expr('%', lhs, self)
    #
    # def __rand__(self, lhs):
    #     return Expr('&', lhs, self)
    #
    # def __rxor__(self, lhs):
    #     return Expr('^', lhs, self)
    #
    # def __ror__(self, lhs):
    #     return Expr('|', lhs, self)
    #
    # def __rrshift__(self, lhs):
    #     return Expr('>>', lhs, self)
    #
    # def __rlshift__(self, lhs):
    #     return Expr('<<', lhs, self)
    #
    # def __rtruediv__(self, lhs):
    #     return Expr('/', lhs, self)
    #
    # def __rfloordiv__(self, lhs):
    #     return Expr('//', lhs, self)
    #
    # def __rmatmul__(self, lhs):
    #     return Expr('@', lhs, self)

    def __call__(self, *args):
        """Call: if 'f' is a Symbol, then f(0) == Expr('f', 0)."""
        if self.args:
            raise ValueError('Can only do a call for a Symbol, not an Expr')
        else:
            return Expr(self.op, *args)

    # Equality and repr
    def __eq__(self, other):
        """x == y' evaluates to True or False; does not build an Expr."""
        return isinstance(other, Expr) and self.op == other.op and self.args == other.args

    def __lt__(self, other):
        return isinstance(other, Expr) and str(self) < str(other)

    def __hash__(self):
        return hash(self.op) ^ hash(self.args)

    def __repr__(self):
        op = self.op
        args = [str(arg) for arg in self.args]
        if op.isidentifier():  # f(x) or f(x, y)
            return '{}({})'.format(op, ', '.join(args)) if args else op
        elif len(args) == 1:  # -x or -(x + 1)
            return op + args[0]
        else:  # (x - y)
            opp = (' ' + op + ' ')
            return '(' + opp.join(args) + ')'


Number = (int, float, complex)
Expression = (Expr, Number)  # An 'Expression' is either an Expr or a Number.


def Symbol(name):
    # a Symbol is an Expr with no args
    return Expr(name)


def symbols(names):
    # return a tuple of Symbols; names is a comma/whitespace delimited str
    return tuple(Symbol(name) for name in names.replace(',', ' ').split())


def subexpressions(x):
    # Yield the subexpressions of an Expression (including x itself)
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpressions(arg)


def arity(expression):
    # number of sub-expressions in this expression
    if isinstance(expression, Expr):
        return len(expression.args)
    else:  # expression is a number
        return 0


# For operators that are not defined in Python


class PartialExpr:

    def __init__(self, op, lhs):
        self.op, self.lhs = op, lhs

    def __or__(self, rhs):
        return Expr(self.op, self.lhs, rhs)

    def __repr__(self):
        return "PartialExpr('{}', {})".format(self.op, self.lhs)


def expr(x):
    return eval(expr_handle_infix_ops(x), defaultkeydict(Symbol)) if isinstance(x, str) else x


infix_ops = '==> <== <=>'.split()


def expr_handle_infix_ops(x):
    # given a str, return a new str with ==> replaced by |'==>'|, etc.

    for op in infix_ops:
        x = x.replace(op, '|' + repr(op) + '|')
    return x


class defaultkeydict(collections.defaultdict):
    # Like defaultdict, but the default_factory is a function of the key.

    def __missing__(self, key):
        self[key] = result = self.default_factory(key)
        return result


class hashabledict(dict):
    # allows hashing by representing a dictionary as tuple of key:value pairs

    def __hash__(self):
        return 1


class Bool(int):
    __str__ = __repr__ = lambda self: 'T' if self else 'F'


T = Bool(True)
F = Bool(False)
