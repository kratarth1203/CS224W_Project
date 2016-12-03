from __future__ import print_function
from builtins import range
import sys, os
import h2o
#from tests import pyunit_utils
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
import cPickle
import numpy as np

def anomaly():
  print("Deep Learning Anomaly Detection MNIST")
  h2o.init(force_connect = False)
  train = h2o.import_file("/afs/.ir/users/k/r/kratarth/private/acads/cs224w/project/CS224W_Project/data/dnodaFeats.txt")
  test = h2o.import_file("/afs/.ir/users/k/r/kratarth/private/acads/cs224w/project/CS224W_Project/data/dnodaFeats.txt")

  predictors = list(range(0,16))
  #resp = 784

  # unsupervised -> drop the response column (digit: 0-9)
  #train = train[predictors]
  #test = test[predictors]

  # 1) LEARN WHAT'S NORMAL
  # train unsupervised Deep Learning autoencoder model on train_hex

  ae_model = H2OAutoEncoderEstimator(activation="Tanh", hidden=[2], l1=1e-5, ignore_const_cols=False, epochs=1)
  ae_model.train(x=predictors,training_frame=train)

  # 2) DETECT OUTLIERS
  # anomaly app computes the per-row reconstruction error for the test data set
  # (passing it through the autoencoder model and computing mean square error (MSE) for each row)
  test_rec_error = ae_model.anomaly(test)

  # 3) VISUALIZE OUTLIERS
  # Let's look at the test set points with low/median/high reconstruction errors.
  # We will now visualize the original test set points and their reconstructions obtained
  # by propagating them through the narrow neural net.

  # Convert the test data into its autoencoded representation (pass through narrow neural net)
  test_recon = ae_model.predict(test)
  h2o.export_file(test_recon, '../data/recon.csv', force = False, parts = 1)
  f = open('../data/recon.csv','r')
  lines = f.readlines()
  f.close()
  lines = lines[1:]
  data = []
  for line in lines:
    words = line.split(',')
    try:
      data.append([float(_) for _ in words])
    except:
      pass
  data = np.array(data, dtype=np.float32)
  mse = np.mean(data, axis = 1)
  epoch_mse = {}
  for i, item in enumerate(mse):
    if (i/54 + 1) in epoch_mse.keys():
      epoch_mse[i/54 + 1].append((item, i))
    else:
      epoch_mse[i/54 + 1] = [(item, i)]
  for k in epoch_mse.keys():
    mse_ = epoch_mse[k]
    sorted_mse = sorted(mse_,key = lambda x:(x[0], x[1]))
    for item, ct in sorted_mse[:3]:
      print('epoch = ', k , ' sensor = ' , ct%54, ' with mse = ', item)

anomaly()

