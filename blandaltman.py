import numpy as np
import pyCompare
import pandas as pd 
import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
import warnings
from scipy import stats

df = pd.read_csv("Results_Final_Main.csv")
# df = pd.read_csv("Results_Final_Main_Outlier.csv") ### Uncomment to analyze videos after removing outliers
#print(df)
df_v = df.to_numpy()
df_v = np.asarray(df_v).astype(float)



def detrendFun(method, data1, data2):
	"""
	Model and remove a mutiplicative offset between data1 and data2 by method

	:param method: Detrending method to use 
	:type method: None or str
	:param numpy.array data1: Array of first measures
	:param numpy.array data2: Array of second measures
	"""

	slope = slopeErr = None

	if method is None:
		pass
	elif method.lower() == 'linear':
		reg = stats.linregress(data1, data2)

		slope = reg.slope
		slopeErr = reg.stderr

		data2 = data2 / slope

	elif method.lower() == 'odr':
		from scipy import odr

		def f(B, x):
			return B[0]*x + B[1]
		linear = odr.Model(f)

		odrData = odr.Data(data1, data2, wd=1./numpy.power(numpy.std(data1),2), we=1./numpy.power(numpy.std(data2),2))

		odrModel = odr.ODR(odrData, linear, beta0=[1., 2.])

		myoutput = odrModel.run()

		slope = myoutput.beta[0]
		slopeErr = myoutput.sd_beta[0]

		data2 = data2 / slope

	else:
		raise NotImplementedError(f"'{detrend}' is not a valid detrending method.")

	return data2, slope, slopeErr


def calculateConfidenceIntervals(md, sd, n, limitOfAgreement, confidenceInterval, confidenceIntervalMethod):
	"""
	Calculate confidence intervals on the mean difference and limits of agreement.

	Two methods are supported, the approximate method descibed by Bland & Altman, and the exact paired method described by Carket.

	:param float md:
	:param float sd:
	:param int n: Number of paired observations
	:param float limitOfAgreement:
	:param float confidenceInterval: Calculate confidence intervals over this range
	:param str confidenceIntervalMethod: Algorithm to calculate CIs
	"""
	confidenceIntervals = dict()

	if not (confidenceInterval < 99.9) & (confidenceInterval > 1):
		raise ValueError(f'"confidenceInterval" must be a number in the range 1 to 99, "{confidenceInterval}" provided.')

	confidenceInterval = confidenceInterval / 100.

	confidenceIntervals['mean'] = stats.norm.interval(confidenceInterval, loc=md, scale=sd/numpy.sqrt(n))

	if confidenceIntervalMethod.lower() == 'exact paired':

		coeffs = parallelCarkeetCIest(n, confidenceInterval, limitOfAgreement)

		coefInner = coeffs[0]
		coefOuter = coeffs[1]

		confidenceIntervals['upperLoA'] = (md + (coefInner * sd),
										   md + (coefOuter * sd))

		confidenceIntervals['lowerLoA'] = (md - (coefOuter * sd),
										   md - (coefInner * sd))

	elif confidenceIntervalMethod.lower() == 'approximate':

		seLoA = ((1/n) + (limitOfAgreement**2 / (2 * (n - 1)))) * (sd**2)
		loARange = numpy.sqrt(seLoA) * stats.t._ppf((1-confidenceInterval)/2., n-1)

		confidenceIntervals['upperLoA'] = ((md + limitOfAgreement*sd) + loARange,
										   (md + limitOfAgreement*sd) - loARange)

		confidenceIntervals['lowerLoA'] = ((md - limitOfAgreement*sd) + loARange,
										   (md - limitOfAgreement*sd) - loARange)

	else:
		raise NotImplementedError(f"'{confidenceIntervalMethod}' is not an valid method of calculating confidance intervals")
	
	return confidenceIntervals


def blandAltman(sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8, limitOfAgreement=1.96, confidenceInterval=95, confidenceIntervalMethod='approximate', percentage=False, detrend=None, title=None, ax=None, figureSize=(10,7), dpi=72, savePath=None, figureFormat='png', meanColour='#6495ED', loaColour='coral', pointColour='#6495ED'):
	"""
	blandAltman(data1, data2, limitOfAgreement=1.96, confidenceInterval=None, **kwargs)

	Generate a Bland-Altman [#]_ [#]_ plot to compare two sets of measurements of the same value.

	Confidence intervals on the limit of agreement may be calculated using:
	- 'exact paired' uses the exact paired method described by Carkeet [#]_
	- 'approximate' uses the approximate method described by Bland & Altman

	The exact paired method will give more accurate results when the number of paired measurements is low (approx < 100), at the expense of much slower plotting time.

	The *detrend* option supports the following options:
	- ``None`` do not attempt to detrend data - plots raw values
	- 'Linear' attempt to model and remove a multiplicative offset between each assay by linear regression
	- 'ODR' attempt to model and remove a multiplicative offset between each assay by Orthogonal distance regression

	:param data1: First measurement
	:type data1: list like
	:param data1: Second measurement
	:type data1: list like
	:param float limitOfAgreement: Multiple of the standard deviation to plot limit of agreement bounds at (defaults to 1.96)
	:param confidenceInterval: If not ``None``, plot the specified percentage confidence interval on the mean and limits of agreement
	:param str confidenceIntervalMethod: Method used to calculated confidence interval on the limits of agreement
	:type confidenceInterval: None or float
	:param detrend: If not ``None`` attempt to detrend by the method specified
	:type detrend: None or str
	:param bool percentage: If ``True``, plot differences as percentages (instead of in the units the data sources are in)
	:param str title: Title text
	:param matplotlib.axes._subplots.AxesSubplot ax: Matplotlib axis handle - if not `None` draw into this axis rather than creating a new figure
	:param figureSize: Figure size as a tuple of (width, height) in inches
	:type figureSize: (float, float)
	:param int dpi: Figure resolution
	:param str savePath: If not ``None``, save figure at this path
	:param str figureFormat: When saving figure use this format
	:param str meanColour: Colour to use for plotting the mean difference
	:param str loaColour: Colour to use for plotting the limits of agreement
	:param str pointColour: Colour for plotting data points

	.. [#] Altman, D. G., and Bland, J. M. “Measurement in Medicine: The Analysis of Method Comparison Studies” Journal of the Royal Statistical Society. Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317. `JSTOR <https://www.jstor.org/stable/2987937>`_.
	.. [#] Altman, D. G., and Bland, J. M. “Measuring agreement in method comparison studies” Statistical Methods in Medical Research, vol. 8, no. 2, 1999, pp. 135–160. `DOI <https://doi.org/10.1177/096228029900800204>`_.
	.. [#] Carkeet, A. "Exact Parametric Confidence Intervals for Bland-Altman Limits of Agreement" Optometry and Vision Science, vol. 92, no 3, 2015, pp. e71–e80 `DOI <https://doi.org/10.1097/OPX.0000000000000513>`_.
	"""
	if not limitOfAgreement > 0:
		raise ValueError('"limitOfAgreement" must be a number greater than zero.') 

	# Try to coerce variables to numpy arrays
	# data1 = numpy.asarray(data1)
	# data2 = numpy.asarray(data2)
	data1 = np.concatenate((sample_1, sample_2, sample_3, sample_4))
	data2 = np.concatenate((sample_5, sample_6, sample_7, sample_8))
	data1 = numpy.asarray(data1)
	data2 = numpy.asarray(data2)

	data2, slope, slopeErr = detrendFun(detrend, data1, data2)

	mean = numpy.mean([data1, data2], axis=0)

	if percentage:
		diff = ((data1 - data2) / mean) * 100
	else:
		diff = data1 - data2

	md = numpy.mean(diff)
	sd = numpy.std(diff, axis=0)

	if confidenceInterval:
		confidenceIntervals = calculateConfidenceIntervals(md, sd, len(diff), limitOfAgreement, confidenceInterval, confidenceIntervalMethod)

	else:
		confidenceIntervals = dict()

	ax = _drawBlandAltman(mean, diff, md, sd, percentage,
						  limitOfAgreement,
						  confidenceIntervals,
						  (detrend, slope, slopeErr),
						  title,
						  ax,
						  figureSize,
						  dpi,
						  savePath,
						  figureFormat,
						  meanColour,
						  loaColour,
						  pointColour)

	if ax is not None:
		return ax


def _drawBlandAltman(mean, diff, md, sd, percentage, limitOfAgreement, confidenceIntervals, detrend, title, ax, figureSize, dpi, savePath, figureFormat, meanColour, loaColour, pointColour):
	"""
	Sub function to draw the plot.
	"""
	if ax is None:
		fig, ax = plt.subplots(1,1, figsize=figureSize, dpi=dpi)
		plt.rcParams.update({'font.size': 15,'xtick.labelsize':15,
         'ytick.labelsize':15})

		ax.tick_params(axis='x', labelsize=15)
		ax.tick_params(axis='y', labelsize=15)
		# ax.rcParams.update({'font.size': 15})

		# ax=ax[0,0]
		draw = True
	else:
		draw = False

	##
	# Plot CIs if calculated
	##
	if 'mean' in confidenceIntervals.keys():
		ax.axhspan(confidenceIntervals['mean'][0],
				   confidenceIntervals['mean'][1],
				   facecolor='lightblue', alpha=0.2)

	if 'upperLoA' in confidenceIntervals.keys():
		ax.axhspan(confidenceIntervals['upperLoA'][0],
				   confidenceIntervals['upperLoA'][1],
				   facecolor='wheat', alpha=0.2)

	if 'lowerLoA' in confidenceIntervals.keys():
		ax.axhspan(confidenceIntervals['lowerLoA'][0],
				   confidenceIntervals['lowerLoA'][1],
				   facecolor='wheat', alpha=0.2)

	##
	# Plot the mean diff and LoA
	##
	ax.axhline(md, color=meanColour, linestyle='--')
	ax.axhline(md + limitOfAgreement*sd, color=loaColour, linestyle='--')
	ax.axhline(md - limitOfAgreement*sd, color=loaColour, linestyle='--')

	##
	# Plot the data points
	##
	# ax.scatter(mean[0:22], diff[0:22], alpha=0.8,  c='orange', marker='.', s=100, label='India Male')
	# ax.scatter(mean[22:44], diff[22:44], alpha=0.8,  c='blue', marker='.', s=100, label='India Female')
	# ax.scatter(mean[44:66], diff[44:66], alpha=0.8,  c='red', marker='.', s=100, label='Sierra Leone Male')
	# ax.scatter(mean[66:88], diff[66:88], alpha=0.8,  c='purple', marker='.', s=100, label='Sierra Leone Female')
	ax.scatter(mean[0:20], diff[0:20], alpha=0.8,  c='orange', marker='.', s=100, label='India Male')
	ax.scatter(mean[20:39], diff[20:39], alpha=0.8,  c='blue', marker='.', s=100, label='India Female')
	ax.scatter(mean[39:59], diff[39:59], alpha=0.8,  c='red', marker='.', s=100, label='Sierra Leone Male')
	ax.scatter(mean[59:77], diff[59:77], alpha=0.8,  c='purple', marker='.', s=100, label='Sierra Leone Female')
	ax.set_ylim(-50, 70)
	ax.legend(loc='upper right', fontsize=12)
	trans = transforms.blended_transform_factory(
		ax.transAxes, ax.transData)

	limitOfAgreementRange = (md + (limitOfAgreement * sd)) - (md - limitOfAgreement*sd)
	offset = (limitOfAgreementRange / 100.0) * 1.5

	ax.text(0.98, md + offset, 'Mean', ha="right", va="bottom", transform=trans)
	ax.text(0.98, md - offset, f'{md:.2f}', ha="right", va="top", transform=trans)

	ax.text(0.98, md + (limitOfAgreement * sd) + offset, f'+{limitOfAgreement:.2f} SD', ha="right", va="bottom", transform=trans)
	ax.text(0.98, md + (limitOfAgreement * sd) - offset, f'{md + limitOfAgreement*sd:.2f}', ha="right", va="top", transform=trans)

	ax.text(0.98, md - (limitOfAgreement * sd) - offset, f'-{limitOfAgreement:.2f} SD', ha="right", va="top", transform=trans)
	ax.text(0.98, md - (limitOfAgreement * sd) + offset, f'{md - limitOfAgreement*sd:.2f}', ha="right", va="bottom", transform=trans)

	# Only draw spine between extent of the data
	# ax.spines['left'].set_bounds(min(diff), max(diff))
	# ax.spines['bottom'].set_bounds(min(mean), max(mean))

	# Hide the right and top spines
	# ax.spines['right'].set_visible(False)
	# ax.spines['top'].set_visible(False)

	if percentage:
		ax.set_ylabel('Percentage difference between methods', fontsize=20)
	else:
		ax.set_ylabel('Difference between methods', fontsize=20)
	ax.set_xlabel('Mean of methods', fontsize=20)

	# tickLocs = ax.xaxis.get_ticklocs()
	# cadenceX = tickLocs[2] - tickLocs[1]
	# tickLocs = rangeFrameLocator(tickLocs, (min(mean), max(mean)))
	# ax.xaxis.set_major_locator(ticker.FixedLocator(tickLocs))

	# tickLocs = ax.yaxis.get_ticklocs()
	# cadenceY = tickLocs[2] - tickLocs[1]
	# tickLocs = rangeFrameLocator(tickLocs, (min(diff), max(diff)))
	# ax.yaxis.set_major_locator(ticker.FixedLocator(tickLocs))

	# plt.draw() # Force drawing to populate tick labels

	# labels = rangeFrameLabler(ax.xaxis.get_ticklocs(), [item.get_text() for item in ax.get_xticklabels()], cadenceX)
	# ax.set_xticklabels(labels)

	# labels = rangeFrameLabler(ax.yaxis.get_ticklocs(), [item.get_text() for item in ax.get_yticklabels()], cadenceY)
	# ax.set_yticklabels(labels)


	# ax.patch.set_alpha(0)

	if detrend[0] is None:
		pass
	else:
		plt.text(1, -0.1, f'{detrend[0]} slope correction factor: {detrend[1]:.2f} ± {detrend[2]:.2f}', ha='right', transform=ax.transAxes)

	if title:
		ax.set_title(title)

	##
	# Save or draw
	##
	plt.tight_layout()

	if (savePath is not None) & draw:
		fig.savefig(savePath, format=figureFormat, dpi=dpi)
		plt.close()
	elif draw:
		plt.show()
	else:
		return ax


### For Bland Altman plots of complete samples
for i in [4,8,12,16,20]:
	sample_1 = df_v[:,0]
	sample_2 = df_v[:,1]
	sample_3 = df_v[:,2]
	sample_4 = df_v[:,3]
	sample_5 = df_v[:,i]
	sample_6 = df_v[:,i+1]
	sample_7 = df_v[:,i+2]
	sample_8 = df_v[:,i+3]

	name = "Bland_Altman" + str(i) + ".png"
	blandAltman(sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8,
			limitOfAgreement=1.96,
			confidenceInterval=95,
			confidenceIntervalMethod='approximate',
			detrend=None,
			percentage=False,
			savePath=name)


## Uncomment the following lines (for loop) for Bland Altman plots after removing outliers
# for i in [4,8,12,16,20]:
# 	sample_1 = df_v[0:20,0]
# 	sample_2 = df_v[0:19,1]
# 	sample_3 = df_v[0:20,2]
# 	sample_4 = df_v[0:18,3]
# 	sample_5 = df_v[0:20,i]
# 	sample_6 = df_v[0:19,i+1]
# 	sample_7 = df_v[0:20,i+2]
# 	sample_8 = df_v[0:18,i+3]

# 	name = "Bland_Altman_outlier" + str(i) + ".png"  ### Uncomment for Bland Altman plots after removing outliers
# 	blandAltman(sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8,
# 			limitOfAgreement=1.96,
# 			confidenceInterval=95,
# 			confidenceIntervalMethod='approximate',
# 			detrend=None,
# 			percentage=False,
# 			savePath=name)


	
	