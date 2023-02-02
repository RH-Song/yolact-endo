import re, sys, os
import matplotlib.pyplot as plt

from utils.functions import MovingAverage

with open(sys.argv[1], 'r') as f:
	inp = f.read()
{"type": "train", "session": 0, "data": {"loss": {"B": 2.70649, "M": 2.56656, "C": 2.21814, "S": 0.19761, "T": 7.6888}, "epoch": 1, "iter": 442, "lr": 0.0008956, "elapsed": 0.38335442543029785}, "time": 1609144996.3312762}

patterns = {
	'train': re.compile(r'\[\s*(?P<epoch>\d+)\]\s*(?P<iteration>\d+) \|\| B: (?P<b>\S+) \| C: (?P<c>\S+) \| M: (?P<m>\S+) \|( S: (?P<s>\S+) \|)? T: (?P<t>\S+)'),
	'val': re.compile(r'\s*(?P<type>[a-z]+) \|\s*(?P<all>\S+)')
}
data = {key: [] for key in patterns}

for line in inp.split('\n'):
	for key, pattern in patterns.items():
		f = pattern.search(line)
		
		if f is not None:
			datum = f.groupdict()
			for k, v in datum.items():
				if v is not None:
					try:
						v = float(v)
					except ValueError:
						pass
					datum[k] = v
			
			if key == 'val':
				datum = (datum, data['train'][-1])
			data[key].append(datum)
			break


def smoother(y, interval=100):
	avg = MovingAverage(interval)

	for i in range(len(y)):
		avg.append(y[i])
		y[i] = avg.get_avg()
	
	return y

def plot_train(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Training Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	loss_names = ['BBox Loss', 'Conf Loss', 'Mask Loss']

	x = [x['iteration'] for x in data]
	plt.plot(x, smoother([y['b'] for y in data]))
	plt.plot(x, smoother([y['c'] for y in data]))
	plt.plot(x, smoother([y['m'] for y in data]))

	# if data[0]['s'] is not None:
	# 	plt.plot(x, smoother([y['s'] for y in data]))
	# 	loss_names.append('Segmentation Loss')

	plt.legend(loss_names)
	plt.show()

def plot_val(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Validation mAP')
	plt.xlabel('Epoch')
	plt.ylabel('mAP')

	x = [x[1]['epoch'] for x in data if x[0]['type'] == 'box']
	plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'box'])
	plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'mask'])

	plt.legend(['BBox mAP', 'Mask mAP'])
	plt.show()

if len(sys.argv) > 2 and sys.argv[2] == 'val':
	plot_val(data['val'])
else:
	plot_train(data['train'])
