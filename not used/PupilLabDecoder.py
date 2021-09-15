import msgpack
import json
import file_methods

filepath = "E:/RITEyes/Raw Data/Ambar/1/Gaze/calibrations/ManualCalibration-21524841-d3cc-423d-b0ec-068c682fb15f.plcal"



with open(filepath, "rb") as data_file:
	byte_data= data_file.read()

data_loaded = msgpack.unpackb(byte_data, use_list=False, strict_map_key=False)
json_str = json.dumps(data_loaded, indent = 4)

# plfile = file_methods.load_pldata_file("E:/RITEyes/Raw Data/Ambar/1/Gaze", "notify")
# json_str = json.dumps(plfile, indent = 4)
print(json_str)

# print
#print(json_str)