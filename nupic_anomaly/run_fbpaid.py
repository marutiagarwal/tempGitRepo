#!/usr/bin/env python

# ----------------------------------------------------------------------
# https://github.com/numenta/nupic/wiki/Online-Prediction-Framework
# https://github.com/numenta/nupic/blob/master/src/nupic/frameworks/opf/common_models/anomaly_params_random_encoder/best_single_metric_anomaly_params_tm_cpp.json
# https://github.com/numenta/nupic/wiki/Models
# https://github.com/numenta/nupic/wiki/NuPIC-Input-Data-File-Format

# -----------------------------------------------------------------------
# Anomaly Parameters
# /usr/local/lib/python2.7/dist-packages/nupic/encoders/scalar.py

# -----------------------------------------------------------------------
# https://discourse.numenta.org/t/is-my-data-being-predicted-correctly/735/14
# https://discourse.numenta.org/t/why-am-i-seeing-lot-of-false-positives/810/2
# ----------------------------------------------------------------------
# SWARMS
# run a swarm to generate the model parameter file
# https://github.com/numenta/nupic/wiki/Running-Swarms
# https://discourse.numenta.org/t/understanding-nupic-and-troubleshooting-to-get-the-best-results/962
# ----------------------------------------------------------------------

# Compute likelihood that the anomaly score comes from the same data distribution as 
# the previous anomaly scores

"""
Groups together code used for creating a NuPIC model and dealing with IO.
(This is a component of the One Hot Gym Anomaly Tutorial.)
"""
import importlib
import sys
import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory

import nupic_anomaly_output

from time import time
from cPickle import load, dump, HIGHEST_PROTOCOL

DESCRIPTION = (
  "Starts a NuPIC model from the model params returned by the swarm\n"
  "and pushes each line of input from the gym into the model. Results\n"
  "are written to an output file (default) or plotted dynamically if\n"
  "the --plot option is specified.\n"
)

DATA_DIR = "."
MODEL_PARAMS_DIR = "./model_params"
# '7/2/10 0:00'
DATE_FORMAT = "%m/%d/%y %H:%M"
FEAT_NAME = "post_like"
GYM_NAME = "fb_paid_" + FEAT_NAME + "_hourly"

def createModel(modelParams):
  """
  Given a model params dictionary, create a CLA Model. Automatically enables
  inference for FEAT_NAME.
  :param modelParams: Model params dict
  :return: OPF Model object
  """
  t0 = time()
  model = ModelFactory.create(modelParams)
  model.enableLearning()  
  model.enableInference({"predictedField": FEAT_NAME})
  print 'time taken in creating model = ',time()-t0

  # t0 = time()
  # model.save('/home/magarwal/logoDetective/core/anomalyDetection/nupic/nupic/examples/opf/clients/hotgym/anomaly/one_gym/model_save/model.pkl')
  # print 'time taken in saving model = ',time()-t0
  return model



def getModelParamsFromName(gymName):
  """
  Given a gym name, assumes a matching model params python module exists within
  the model_params directory and attempts to import it.
  :param gymName: Gym name, used to guess the model params module name.
  :return: OPF Model params dictionary
  """
  importName = "model_params.%s_model_params" % (
    gymName.replace(" ", "_").replace("-", "_")
  )
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. Run swarm first!"% gymName)
  return importedModelParams



def runIoThroughNupic(inputData, model, gymName, plot):
  """
  Handles looping over the input data and passing each row into the given model
  object, as well as extracting the result object and passing it into an output
  handler.
  :param inputData: file path to input data CSV
  :param model: OPF Model object
  :param gymName: Gym name, used for output handler naming
  :param plot: Whether to use matplotlib or not. If false, uses file output.
  """
  # t0 = time()
  # # model = load(open('model.pkl', 'rb'))
  # model = ModelFactory.loadFromCheckpoint('/home/magarwal/logoDetective/core/anomalyDetection/nupic/nupic/examples/opf/clients/hotgym/anomaly/one_gym/model_save/model.pkl')
  # print 'time taken in loading model = ',time()-t0

  inputFile = open(inputData, "rb")
  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  shifter = InferenceShifter()
  if plot:
    output = nupic_anomaly_output.NuPICPlotOutput(gymName)
  else:
    output = nupic_anomaly_output.NuPICFileOutput(gymName)

  counter = 0
  for row in csvReader:
    counter += 1
    if (counter % 1000 == 0):
      print "Read %i lines..." % counter
    timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
    feat = float(row[1])
    result = model.run({"time_start": timestamp, FEAT_NAME: feat})
    if plot:
      result = shifter.shift(result)

    # print 'result = ',result
    prediction = result.inferences["multiStepBestPredictions"][1]
    anomalyScore = result.inferences["anomalyScore"]
    output.write(timestamp, feat, prediction, anomalyScore)

  inputFile.close()
  output.close()



def runModel(gymName, plot=False):
  """
  Assumes the gynName corresponds to both a like-named model_params file in the
  model_params directory, and that the data exists in a like-named CSV file in
  the current directory.
  :param gymName: Important for finding model params and input CSV file
  :param plot: Plot in matplotlib? Don't use this unless matplotlib is
  installed.
  """
  print "Creating model from %s..." % gymName
  model = createModel(getModelParamsFromName(gymName))
  inputData = "%s/%s.csv" % (DATA_DIR, gymName.replace(" ", "_"))
  runIoThroughNupic(inputData, model, gymName, plot)



if __name__ == "__main__":
  print DESCRIPTION
  plot = False
  args = sys.argv[1:]
  if "--plot" in args:
    plot = True
  runModel(GYM_NAME, plot=plot)
