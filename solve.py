from sympy import *
from sympy.solvers.solveset import linsolve
import sys

# --- 2d ---
# P, T, R = phi, theta, r of coordinate system we're sitting in
# x0, y0, z0 = vertex of track
# p0, t0, rho = phi, theta, r of track, where rt is the track parameter that we're solving for!
# x0, y0, z0, p0, t0 are all constant!
x0, y0, z0, P, T, R, p0, t0, rho = symbols('x0 y0 z0 P T R p0 t0 rho')

##
#x = Eq(x0 + rho*cos(p0) - R*cos(P))
#y = Eq(y0 + rho*sin(p0) - R*sin(P))
##
#
## rho(Phi)
#R_solved = solve(x,R)
#print 'R = ',R_solved
#y2=y.subs({R:R_solved[0]})
#rho_solved = solve(y2,rho)
#print 'rho(Phi) = ',rho_solved
#
## rho(R)
#P_solved = solve(x,P)
#print 'Phi = ',P_solved
#y2=y.subs({P:P_solved[0]})
#rho_solved = solve(y2,rho)
#print 'rho(R) = ',rho_solved

# --- 3d ---
print '\n3D:\n'
#
x = Eq(x0 + rho*cos(p0)*sin(t0) - R*cos(P)*sin(T))
y = Eq(y0 + rho*sin(p0)*sin(t0) - R*sin(P)*sin(T))
z = Eq(z0 + rho*cos(t0) - R*cos(T))
#

# rho(Phi)
#T_solved = solve(z,T,check=False)
#print 'T = ',T_solved
#y2 = y.subs({T:T_solved[0]})
#y2 = trigsimp(y2)
#y2 = simplify(y2)
#print y2
#R_solved = solve(y2,R,check=False)
#print 'R = ',R_solved
#x2 = x.subs({T:T_solved[0]})
#x2 = trigsimp(x2)
#x2 = simplify(x2)
##print z2
#x3 = x2.subs({R:R_solved[1]})
#x3 = trigsimp(x3)
#x3 = simplify(x3)
#print x3
#rho_solved = solve(x3,rho,check=False)
#rho_solved = trigsimp(rho_solved)
#print 'rho(P) = ',rho_solved

# rho(Theta)
#P_solved = solve(x,P,check=False)
#print 'P = ',P_solved
#y2 = y.subs({P:P_solved[1]})
#y2 = trigsimp(y2)
#y2 = simplify(y2)
#print y2
#R_solved = solve(y2,R,check=False)
#print 'R = ',R_solved
#z2 = z.subs({P:P_solved[1]})
#z2 = trigsimp(z2)
#z2 = simplify(z2)
##print z2
#z3 = z2.subs({R:R_solved[1]})
#z3 = trigsimp(z3)
#z3 = simplify(z3)
#print z3
#rho_solved = solve(z3,rho,check=False)
#rho_solved = trigsimp(rho_solved[0])
#print 'rho(T) = ',rho_solved

# rho(R)
#P_solved = solve(x,P,check=False)
#print 'P = ',P_solved
#y2 = y.subs({P:P_solved[1]})
#y2 = trigsimp(y2)
#y2 = simplify(y2)
#print y2
#T_solved = solve(y2,T,check=False)
#print 'T = ',T_solved
#z2 = z.subs({P:P_solved[1]})
#z2 = trigsimp(z2)
#z2 = simplify(z2)
##print z2
#z3 = z2.subs({T:T_solved[2]})
#z3 = trigsimp(z3)
#z3 = simplify(z3)
#print z3
#rho_solved = solve(z3,rho,check=False)
#rho_solved = trigsimp(rho_solved[0])
#print 'rho(R) = ',rho_solved

#m = Eq((rho*sin(t0)**2+x0*sin(t0)*cos(p0)+y0*sin(t0)*sin(p0))*(z0+rho*cos(t0))-cos(t0)*((x0+rho*sin(t0)*cos(p0))**2+(y0+rho*sin(t0)*sin(p0))**2))
m = Eq( cos(t0)*((x0+rho*sin(t0)*cos(p0))**2 + (y0+rho*sin(t0)*sin(p0))**2 + (z0+rho*cos(t0))**2) - (z0 + rho*cos(t0))*((x0+rho*sin(t0)*cos(p0))*sin(t0)*cos(p0) + (y0+rho*sin(t0)*sin(p0))*sin(t0)*sin(p0) + (z0+rho*cos(t0))*cos(t0) ))
m_solved = solve(m,rho,check=False)
#m_solved = trigsimp(m_solved[1]) 
print m_solved

