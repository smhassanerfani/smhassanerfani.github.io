# Symbolic Solution with SymPy
import sympy as sp

# Define variables
t = sp.Symbol("t")
y = sp.Function("y")(t)
print(y)  # y(t)

# Define the ODE using sp.Eq
diff_eq = sp.Eq(y.diff(t), 2*t - y)
print(diff_eq)  # Eq(Derivative(y(t), t), 2*t - y(t))

# Check left and right-hand sides
print(diff_eq.lhs)  # Derivative(y(t), t)
print(diff_eq.rhs)  # 2*t - y(t)

# Solve ODE without initial conditions
sol = sp.dsolve(diff_eq, y)
print(sol)  # Eq(y(t), C1*exp(-t) + 2*t - 2)

# Solve with initial condition y(0) = -1
ics = {y.subs(t, 0): -1}
ivp = sp.dsolve(diff_eq, y, ics=ics)
print(ivp)  # Eq(y(t), 2*t - 2 + exp(-t))
