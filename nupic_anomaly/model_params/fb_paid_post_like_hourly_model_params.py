MODEL_PARAMS = \
{u'aggregationInfo': {u'days': 0,
                      u'fields': [],
                      u'hours': 0,
                      u'microseconds': 0,
                      u'milliseconds': 0,
                      u'minutes': 0,
                      u'months': 0,
                      u'seconds': 0,
                      u'weeks': 0,
                      u'years': 0},
 u'model': u'CLA',
 u'modelParams': {u'anomalyParams': {u'anomalyCacheRecords': None,
                                     u'autoDetectThreshold': None,
                                     u'autoDetectWaitRecords': 5030},
                  # u'clEnable': False,
                  u'clParams': {u'alpha': 0.035828933612158,
                                u'regionName': u'SDRClassifierRegion',
                                u'steps': u'1',
                                u'verbosity': 0},
                  u'inferenceType': u'TemporalAnomaly',
                  u'sensorParams': {u'encoders': { '_classifierInput': { 'classifierOnly': True,
                                                                         'clipInput': True,
                                                                         'fieldname': 'post_like',
                                                                         'maxval': 899.0,
                                                                         'minval': 1.0,
                                                                         'n': 115,     #n - number of bits in the representation (must be > w)
                                                                         'name': '_classifierInput',
                                                                         'type': 'ScalarEncoder',
                                                                         'w': 21},
                                                               u'post_like': { 'clipInput': True,
                                                                         'fieldname': 'post_like',
                                                                         'maxval': 899.0,
                                                                         'minval': 1.0,
                                                                         'n': 115,
                                                                         'name': 'post_like',
                                                                         'type': 'ScalarEncoder',
                                                                         'w': 21}},
                                    u'sensorAutoReset': None,
                                    u'verbosity': 0},
                  u'spEnable': True,
                  u'spParams': {u'columnCount': 2048,
                                u'globalInhibition': 1,
                                u'inputWidth': 0,
                                u'maxBoost': 1.0,
                                u'numActiveColumnsPerInhArea': 40,
                                u'potentialPct': 0.8,
                                u'seed': 1956,
                                u'spVerbosity': 0,
                                u'spatialImp': u'cpp',
                                u'synPermActiveInc': 0.003,
                                u'synPermConnected': 0.2,
                                u'synPermInactiveDec': 0.0005},
                  u'tpEnable': True,
                  u'tpParams': {u'activationThreshold': 13,
                                u'cellsPerColumn': 32,
                                u'columnCount': 2048,
                                u'globalDecay': 0.0,
                                u'initialPerm': 0.21,
                                u'inputWidth': 2048,
                                u'maxAge': 0,
                                u'maxSegmentsPerCell': 128,
                                u'maxSynapsesPerSegment': 32,
                                u'minThreshold': 10,
                                u'newSynapseCount': 20,
                                u'outputType': u'normal',
                                u'pamLength': 3,
                                u'permanenceDec': 0.1,
                                u'permanenceInc': 0.1,
                                u'seed': 1960,
                                u'temporalImp': u'cpp',
                                u'verbosity': 0},
                  u'trainSPNetOnlyIfRequested': False},
 u'predictAheadTime': None,
 u'version': 1}