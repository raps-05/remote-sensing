from flask import Flask,render_template, request,jsonify,redirect
import psycopg2
import pandas as pd
import numpy as np
import datacube
from deafrica_tools.plotting import rgb, display_map
import datacube
import odc.algo
import matplotlib.pyplot as plt
from datacube.utils.cog import write_cog
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.plotting import display_map, rgb
import io
import base64
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import leaflet.pm;
# import leaflet.pm/dist/leaflet.pm.css;
# global area_per_pixel


#display_map(x=lon_range, y=lat_range)



 
app = Flask(__name__)

  

image_base64=''
@app.route('/',methods=['POST','GET'])
def hello_world():
	
	# content_type = request.headers.get('Content-Type')
	# print(content_type," ----- --- -- - -- * - * - ")
#    (content_type == 'application/json'):
	

	return render_template("image.html")

	# return render_template('image.html',img_base64=image_base64)
@app.route('/about', methods=['POST','GET'])
def process():

	return render_template('about.html')
		
@app.route('/form',methods=['POST','GET'])
def form():
	image_base64=''
	
	total_area_array=[]
	pl_a=[]
	ml_a=[]
	ofm_array=[]
	dfm_array=[]
	sfm_array=[]

	# content_type = request.headers.get('Content-Type')
	# print(content_type," ----- --- -- - -- * - * - ")
#    (content_type == 'application/json'):
	if request.method=='POST' :
		
		st= request.form.get('start-date')
		en = request.form.get('end-date')
		op = request.form.get('option')
		x = request.form.get('x')
		y = request.form.get('y')
		print(st,en,op,x,y)
		
		x_ti= json.loads(x)
		y_ti = json.loads(y)
		
		
		print(x_ti)
		
		print(y_ti)
		
		
		for i in range(len(x_ti)):
			
			
			print(x_ti[i][0],y_ti[i][0])
			print(x_ti[i][1],y_ti[i][1])
			# print(i,x[i][0])
			# print(x[i],x[i][0])
			
			lon_range = (x_ti[i][0],y_ti[i][0])
			lat_range = (x_ti[i][1],y_ti[i][1])
		# ------------------------------------
		# lat_range = (15.65, 15.95)
		# lon_range = (80.75, 81.05)
			print(lon_range)
			print(lat_range)
			time_range = (st,en)
			# display_map(x=lon_range, y=lat_range)
			dc = datacube.Datacube(app="04_Plotting")
			ds = dc.load(product="s2a_sen2cor_granule",
							measurements=["red","blue","green","nir","red_edge_2","red_edge_3","nir_2","swir_2","swir_1"],

						x=lon_range,
						y=lat_range,
						time=time_range,
						output_crs='EPSG:6933',
						resolution=(-30, 30))

			print(ds)
		

# Get the spatial resolution of the dataset
			spatial_resolution = np.abs(ds.geobox.affine.a)

	# Calculate the area per pixel
			area_per_pixel = spatial_resolution**2

	# Determine the number of pixels in the dataset
			num_pixels = ds.sizes['x'] * ds.sizes['y']

	# Calculate the total area
			total_area = area_per_pixel * num_pixels


			total_area_km2=total_area/1000000
			total_area_array.append(total_area_km2)



			print("Area per pixel: {} square meters".format(area_per_pixel))
			print("Total area: {} square meters".format(total_area))
			print("Total area: {} square kms".format(total_area_km2))
			dataset =  odc.algo.to_f32(ds)
			if not ds:
				print("values:object")
				al_m='''
				The error can be :
				1. The range of coordinates are not applicable
				2. No data found in the range of dates
				
				Use high range date
				donot go beyond the range of map'''
				return render_template("image.html" ,al_m=al_m)
			# ds=calculate_indices(ds,op,satellite_mission='s2')
	
			if op=='NDVI':
				ds=calculate_indices(ds,op,satellite_mission='s2')
				
				ds_index = ds.NDVI
				plt.figure()
				ds_index.plot()
				plt.xlabel('Value')
				plt.ylabel('Frequency')
				plt.title('Histogram')
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
			

				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()

				print('ndvi')
			
			
			

		# print(ndvi)
		# Generate the plot
		# for i in range(len(ds_index)):
			# plt.figure()
			ds=calculate_indices(ds,'NDVI',satellite_mission='s2')
			ds_index=ds.NDVI
			dense_forest_mask = np.where((ds_index > 0.6) & (ds_index < 0.8), 1, 0)
			open_forest_mask = np.where((ds_index > 0.3) & (ds_index < 0.6) , 1, 0)
			x=np.where((ds_index>0.8) | (ds_index<0.1),1,0)

			sparse_forest_mask = np.where((ds_index > 0.1) & (ds_index < 0.3) , 1, 0)
			f_a=[]
			
			w=np.sum(x[0])
			print(np.sum(x[0]),"---",area_per_pixel)
			ta=area_per_pixel*w
			ta2=ta/1000000
			print(ta,ta2)
			time_values = ds_index.time.values
			d=['time','dfm','ofm','sfm','tfa']
			x=[]
			for i in range(len(dense_forest_mask)):
				w=[]
				w.append(pd.to_datetime(time_values[i]))
				w.append(area(dense_forest_mask[i],area_per_pixel))
				w.append(area(open_forest_mask[i],area_per_pixel))
				w.append(area(sparse_forest_mask[i],area_per_pixel))
				w.append(area(dense_forest_mask[i],area_per_pixel)+area(open_forest_mask[i],area_per_pixel)+area(sparse_forest_mask[i],area_per_pixel))
				f_a.append(area(dense_forest_mask[i],area_per_pixel)+area(open_forest_mask[i],area_per_pixel)+area(sparse_forest_mask[i],area_per_pixel))
				x.append(w)
	# df['time'] = pd.to_datetime(time_values)
			df = pd.DataFrame(x, columns=d)
			print(df)

# Assuming your time column is named 'time' and the value column is named 'ndvi'
# Convert the 'time' column to pandas timetime if it's not already in that format
# Read the CSV file into a pandas DataFrame

			df['time'] = pd.to_datetime(df['time'])
			print(df.head())
			X_train=df['time']
			y_train=df['ofm']
	# Split the data into training and test sets
	# X_train, X_test, y_train, y_test = train_test_split(df['time'], df['dfm'],test_size=0.3,shuffle=False)

	# Extract the time components as features
			X_train_features = pd.DataFrame()
			X_train_features['year'] = X_train.dt.year
			X_train_features['month'] = X_train.dt.month
			X_train_features['day'] = X_train.dt.day
			# Add more features as per your requirements

			# Initialize and fit the Random Forest Regressor model
			model = RandomForestRegressor()
			model.fit(X_train_features, y_train)

			# Extract features from the test data
			X_test_features = pd.DataFrame()
			X_test_features['year'] = [2018]
			X_test_features['month'] =[ 5]
			X_test_features['day'] = [5]
			# Add more features as per your requirements
			print(X_train_features.head())
			print(X_test_features.head())
			# Predict the values
			predictions = model.predict(X_test_features)

			# Print the predictions
			print(predictions)
			prede=model.predict(X_train_features)
			print(prede)
		
			
			if op=='NDWI':
				ds=calculate_indices(ds,op,satellite_mission='s2')
				
				ds_index = ds.NDWI
				plt.figure(figsize=(10, 9))
				ds_index.plot()
				plt.xlabel('Value')
				plt.ylabel('Frequency')
				plt.title('Histogram')
				buffer = io.BytesIO()
				plt.savefig(buffer, format='png')
				buffer.seek(0)
			

				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				

				print('ndwi')
				buffer.close()
			plt.figure(figsize=(10, 9))
			ds_index.plot(col='time', vmin=-1, vmax=1)

			# Convert the plot to a PNG image in memory
			buffer = io.BytesIO()
			plt.savefig(buffer, format='png')
			buffer.seek(0)
			image_base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
			buffer.close()
			indices = np.arange(len(X_train))
			print("indices : : : ===  ",indices)

			
			if op=='ML PREDICTION':


			
			
				plt.figure()
				plt.plot(indices, df['ofm'], color='blue', label='Actual')
			
				plt.plot(indices, prede, color='red', label='Predicted')
			

				# Add labels and title
				plt.xlabel('Data Point')
				plt.ylabel('Value')
				plt.title('Random Forest Predictions - Actual vs. Predicted')

				# Add legend
				plt.legend()
				buffer=io.BytesIO()
				plt.savefig(buffer, format='png')

				buffer.seek(0)
				image_base64_2=base64.b64encode(buffer.read()).decode('utf-8')
				buffer.close()
				
		

				print('ndwi')
				buffer.close()
			
			
			ml_a.append(image_base64_2)
			pl_a.append(image_base64_1)
	
	print(len(pl_a),len(ml_a))
	
	return jsonify({'pl_a': pl_a,'ml_a':ml_a,'totala':total_area_km2})
def area( a ,area_per_pixel):
	print(np.sum(a))
	xw=area_per_pixel*np.sum(a)
	print(xw)
	return xw/1000000

def image_to_base64(image):
    plt.figure(figsize=(8,8))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return image_base64


	
if __name__ == '__main__':
	app.run(debug=True,port=7000)