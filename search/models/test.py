"""
A module to get some code coverage on the deephyper.search.models module.
If this file is run and nothing is output you're good to go.
"""
from deephyper.search.models.base import param, step, prior, drt, SKOptParser

def main():
    # Construct some parameters and test their methods and fields.
     a = param.continuous("a", 1, 2)
     assert(a.name == "a")
     assert(a.low == 1.0)
     assert(a.high == 2.0)
     assert(a.prior == prior.UNIFORM)

     b = param.continuous("b", 23, 24, prior=prior.LOGUNIFORM)
     assert(b.prior == prior.LOGUNIFORM)

     c = param.discrete("c", 0, 10)
     assert(c.name == "c")
     assert(c.low == 0)
     assert(c.high == 10)
     assert(c.step_type == step.ARITHMETIC)
     assert(c.step_size == 1)
     assert(c.drt == drt.DEFAULT)
     assert(c.map_negative == False)
     assert(c.map_to_interval(c.max_n) == 10)
     assert(c.interval_list == list(range(11)))

     d = param.discrete("d", -10, 0)
     assert(d.map_to_interval(d.max_n) == 0)
     assert(d.interval_list == list(range(-10, 1)))

     e = param.discrete("e", 1, 64, step.GEOMETRIC, 2)
     assert(e.map_to_interval(e.max_n) == 64)
     assert(e.interval_list == [1,2,4,8,16,32,64])

     f = param.discrete("f", 1, 64, step.GEOMETRIC, 2, map_negative=True)
     assert(f.map_to_interval(f.max_n) == -64)
     assert(f.interval_list == [-1, -2, -4, -8, -16, -32, -64])

     g = param.discrete("g", 1, 64, step.GEOMETRIC, 2, drt=drt.ORDINAL)
     assert(g.drt == drt.ORDINAL)

     # Try running the parser.
     space = [a,b,c,d,e,f,g]
     skopt_space = [SKOptParser.transform_param(param) for param in space]

main()
