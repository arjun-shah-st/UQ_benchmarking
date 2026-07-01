import numpy as np 
import pandas as pd 


from UBAS.generators.shift_wing_generator import ShiftWingGenerator 

gen = ShiftWingGenerator(qoi="CD")

input = [ 9.8111,    6.192431, 26.668,     0.5276,    8.124106,  0.85,      4.2041,
 -5.4078  ]

output_x, output_y = gen.generate(input, replace=False, exact_match=True)

print(output_x, output_y)