# DecPIC
A peak deconvolution method was developed based on differential evolution for LC-MS data.

# Required Dependencies

* [ICExtract source](https://github.com/mapancsu/ICExtract)
* [airPLS source](https://github.com/zmzhang/airPLS)
* lwma smooth.
The C++ .dll files of ICExtract, airPLS and lwma methods can be downloaded in [url](https://github.com/mapancsu/ICExtract/releases/tag/DecPIC). 
	
# Usage

* Download DecPIC_python.zip file from [url](https://github.com/mapancsu/DecPIC/releases).
* Upzip DecPIC_python.zip file and go to /DecPIC_python37 directory.
* Run following Python code fragment to extract ion chromatogram from mzXML or mzML file.

	```python
	from CIC_extraction import Get_CIC, Findpeaks, getpeakgroup, DEEMGfit
	CICs = Get_CIC(file, 0.02, 300, 5, 10, 100)
	```
* Run following Python code fragment to find peak information of ion chromatograms.
	```python
	ic_ind = 1500
	ic = CICs[ic_ind][:, 2] ## get ic intensity vector
	xb, xs, noise, total_peakpoint = Findpeaks(ic, 5, 2, 300)  
    infor_group = getpeakgroup(total_peakpoint, xb, thre=0.1)
	```
* Run following Python code fragment to build EMG model for complex ion chromatograms.
	```python
	result = DEEMGfit(xdata, ydata)
	```

# Contact

For any questions, please contact:  [mapan_spy@163.com](mailto:mapan_spy@163.com)
