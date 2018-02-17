# based on 
# https://chrishavlin.wordpress.com/2016/12/01/shapely-polygons-coloring/
# help with
# https://gis.stackexchange.com/questions/74808/extract-a-record-by-name-in-pyshp

import shapefile
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

import os
import pandas as pd 
import numpy as np

import glob
import imageio

"""
 IMPORT THE SHAPEFILE 
"""
shp_file_base = '\\ESRI-PC4-2017R1'
shp_dat_dir = r'D:\data\ESRI_PC4_2017R1_shp' 
val_dat_dir = r'D:\data\trx_gemeente'
plt_dir = r'D:\plot'
sf = shapefile.Reader(shp_dat_dir+shp_file_base)


# feature_CMP = fread("amsterdam_aggr_2015_dec.csv")
# feature_CMP = fread("amsterdam_aggr_201512.csv")
feature_CMP = pd.read_csv(val_dat_dir + '\\amsterdam_aggr_2015.csv')
feature_CMP.date = pd.to_datetime(feature_CMP.date)
feature_CMP = feature_CMP.set_index('date')

os.chdir(plt_dir)

for u_date in feature_CMP.index.unique():
	A = feature_CMP.loc[u_date]
	A['total_count'] = A['total_count']/np.max(A['total_count'])
	A = A.set_index('sel_post')
	plot_area(A, u_date)

	
# Used to create a gif from the generated prediction-plots	

os.chdir(plt_dir)
images = []
filenames = glob.glob("*png")
filenames = np.sort(filenames)[0:100]
for filename in filenames:
	images.append(imageio.imread(filename))

imageio.mimsave('smaller_city.gif', images)

	
def plot_area(dat_input, title):
	plt.figure()
	ax = plt.axes()
	ax.set_aspect('equal')

	icolor = 1
	derp = []
	for shapeRec in sf.iterShapeRecords():
		# pull out shape geometry and records 
		shape=shapeRec.shape
		rec = shapeRec.record
		pc_int = int(rec[1])
		if pc_int not in dat_input.index:
			break
			
		
		# define polygon fill color (facecolor) RGB values:
		R = dat_input.loc[pc_int].total_count	# 1 # (float(icolor)-1.0)/N_ding
		G = 0
		B = 0

		# check number of parts (could use MultiPolygon class of shapely?)
		nparts = len(shape.parts) # total parts
		if nparts == 1:
			polygon = Polygon(shape.points)
			patch = PolygonPatch(polygon, facecolor=[R,G,B], alpha=1.0, zorder=2)
			ax.add_patch(patch)

		else: # loop over parts of each shape, plot separately
			for ip in range(nparts): # loop over parts, plot separately
				i0=shape.parts[ip]
				if ip < nparts-1:
					i1 = shape.parts[ip+1]-1
				else:
					i1 = len(shape.points)

			polygon = Polygon(shape.points[i0:i1+1])
			patch = PolygonPatch(polygon, facecolor=[R,G,B], alpha=1.0, zorder=2)
			ax.add_patch(patch)

		derp.append(shape.bbox)
		icolor = icolor + 1
		plt.xlim(115000, 133000)
		plt.ylim(484000, 495000)
	plt.title(str(title))
	plt.savefig(str(title).replace(':','') + '.png', dpi=250)
	plt.clf()
	plt.cla()
	plt.close()
	#plt.show()
		# plt.xlim(min(np.array(derp)[:,0]), max(np.array(derp)[:,2]))
		# plt.ylim(min(np.array(derp)[:,1]), max(np.array(derp)[:,3]))
	