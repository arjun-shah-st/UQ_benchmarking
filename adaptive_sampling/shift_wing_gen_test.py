import numpy as np 
import pandas as pd 


from UBAS.generators.shift_wing_generator import ShiftWingGenerator 

gen = ShiftWingGenerator(qoi="CD")

input = np.ones((1000, 9)) * np.array([8.8473,35.3156,1.3404,6.4858,4.8286,-4.7202,-3.324,3.7337,0.8])

output_x, output_y = gen.generate(input)

print(output_x, output_y)