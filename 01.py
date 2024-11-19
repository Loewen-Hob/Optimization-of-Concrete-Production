import cplex
print(cplex.__version__)



from docplex.mp.model import Model
mdl = Model(name='test')
mdl.solve()
