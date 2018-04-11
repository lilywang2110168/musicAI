import cPickle as pickle
import gzip
import numpy
from midi_to_statematrix import *

import multi_training
import model
OUTPUT_FOLDER = 'output_classical_music/'
KEYS = {'c':[1,0,1,0,1,1,0,1,0,1,0,1]}
KEY_CORRECT = False
KEY = 'c'

def playsNote(note):
	return not(note[0] == 0 and note[1] == 0)

def gen_adaptive(m,pcs,times,keep_thoughts=False,name="final"):
	xIpt, xOpt = map(lambda x: numpy.array(x, dtype='int8'), multi_training.getPieceSegment(pcs))
	all_outputs = [xOpt[0]]
	if keep_thoughts:
		all_thoughts = []
	m.start_slow_walk(xIpt[0])
	cons = 1
	for time in range(multi_training.batch_len*times):
		resdata = m.slow_walk_fun( cons )

		rescopy = resdata[-1]
		# correct for key
		if KEY_CORRECT:
			for i,note in enumerate(resdata[-1]):
				# note is being played and not in key
				if playsNote(note) and KEYS[KEY][i%12] != 1:
					j = i
					while KEYS[KEY][j%12] != 1:
						j -= 1

					rescopy[i] = [0,0]
					if not(playsNote(rescopy[j-1])): rescopy[j] = note

		nnotes = numpy.sum(rescopy[:, 0])

		if nnotes < 2:
			if cons > 1:
				cons = 1
			cons -= 0.02
		else:
			cons += (1 - cons) * 0.3

		# all_outputs.append(resdata[-1])
		all_outputs.append(rescopy)
		if keep_thoughts:
			all_thoughts.append(rescopy)
	noteStateMatrixToMidi(numpy.array(all_outputs),OUTPUT_FOLDER+name)
	if keep_thoughts:
		pickle.dump(all_thoughts, open(OUTPUT_FOLDER+name+'.p','wb'))

def fetch_train_thoughts(m,pcs,batches,name="trainthoughts"):
	all_thoughts = []
	for i in range(batches):
		ipt, opt = multi_training.getPieceBatch(pcs)
		thoughts = m.update_thought_fun(ipt,opt)
		all_thoughts.append((ipt,opt,thoughts))
	pickle.dump(all_thoughts, open(OUTPUT_FOLDER+name+'.p','wb'))

if __name__ == '__main__':

	pcs = multi_training.loadPieces("classical_music")

	m = model.Model([300,300],[100,50], dropout=0.5)

	#m.learned_config = pickle.load(open("output_jazz/params3000.p", "rb"))

	multi_training.trainPiece(m, pcs, 10000)

	#gen_adaptive(m, pcs, 10, name="composition")

	pickle.dump( m.learned_config, open( "output_classical_music/final_learned_config.p", "wb" ) )
