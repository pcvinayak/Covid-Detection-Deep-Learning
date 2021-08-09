import Augmentor

p = Augmentor.Pipeline("C:\\Users\\pcvin\\Downloads\\study\\dl\\project\\data\\Train")

p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.3, min_factor=1.01, max_factor=1.1)
p.flip_random(probability=0.5)
p.random_brightness(probability=0.5,min_factor=0.7,max_factor=1.3)
p.random_distortion(probability=0.2, grid_width=3,grid_height=3,magnitude=1)
p.shear(probability=0.3,max_shear_left=10,max_shear_right=10)
p.sample(20000,multi_threaded=True)