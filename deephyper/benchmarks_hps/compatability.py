# accept python2 and python3 sets
try:
    from sets import Set as set
except ImportError:
    set = set

# traverse a dictionary in python2 and python3.
def items(_dict):
    try:
        return _dict.iteritems()
    except AttributeError:
        return _dict.items()
